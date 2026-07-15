"""
PASSOS 6 + 7 — Treino do modelo híbrido com validação ROLLING-ORIGIN
(walk-forward), conforme a figura: o treino expande a cada fold e a
validação é SEMPRE 1 dia (val_horizon_steps) imediatamente subsequente
ao último dado treinado. Após validado, o dia entra no treino do fold
seguinte (warm start).

Como cada fold produz uma perda de validação em uma origem temporal
distinta, a média/desvio entre folds JÁ É a validação cruzada temporal
— este módulo funde os passos 6 e 7.

Anti-leak:
  * Janela pertence ao TREINO do fold j somente se ela termina
    (start + backcast + forecast) até o corte c_j.
  * Janela de VALIDAÇÃO é aquela cujo horizonte de forecast cai
    exatamente no dia [c_j, c_j + 24). Seu backcast usa dados <= c_j,
    o que é o comportamento padrão e correto de avaliação rolling-origin:
    o contexto é conhecido, o ALVO (dia seguinte) é inédito.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from config import CFG
from model_hybrid import HybridWLSTMix


class WindowedPTDataset(Dataset):
    """Indexa um .pt gerado nos passos 2-5. Zero processamento no __getitem__."""

    def __init__(self, pt_path):
        pack = torch.load(pt_path, weights_only=False)
        B = CFG.backcast_length
        self.trend_in = pack["trend_norm"][:, :B]
        self.season_in = pack["season_norm"][:, :B]
        self.trend_tg = pack["trend_norm"][:, B:]
        self.season_tg = pack["season_norm"][:, B:]
        # Alvo de classificação = rótulos FUNDIDOS (passo 5) no horizonte.
        self.cls_tg = pack["labels_fused"][:, B:].float()
        self.building_idx = pack["building_idx"]
        self.starts = pack["start"]              # necessário p/ cortes temporais
        self.group = pt_path.stem

    def __len__(self):
        return self.trend_in.shape[0]

    def __getitem__(self, i):
        return {
            "trend_input": self.trend_in[i],
            "season_input": self.season_in[i],
            "trend_target": self.trend_tg[i],
            "season_target": self.season_tg[i],
            "cls_target": self.cls_tg[i],
        }


def load_split(split: str) -> ConcatDataset:
    ds = []
    for sector in CFG.sectors:
        base = CFG.windows_root / split / sector
        if base.exists():
            ds += [WindowedPTDataset(p) for p in sorted(base.glob("*.pt"))]
    if not ds:
        raise RuntimeError(f"Nenhum .pt em {split} — rode os passos 1-5 antes.")
    return ConcatDataset(ds)


# ---------------------------------------------------------------------------
# Construção dos folds rolling-origin (comportamento da figura)
# ---------------------------------------------------------------------------
def rolling_origin_folds(full: ConcatDataset):
    """Gera [(idx_treino, idx_val), ...] com validação de 1 dia por fold.

    Os cortes são POR EDIFÍCIO (séries têm comprimentos diferentes) e
    alinhados ao stride, de modo que exista exatamente uma janela por
    edifício cujo forecast é o dia de validação.
    """
    B, F = CFG.backcast_length, CFG.forecast_length
    day = CFG.val_horizon_steps

    # Indexação global: (idx_global, chave_edifício, start)
    per_building = defaultdict(list)
    offset = 0
    for ds in full.datasets:
        for i in range(len(ds)):
            key = f"{ds.group}:{int(ds.building_idx[i])}"
            per_building[key].append((offset + i, int(ds.starts[i])))
        offset += len(ds)

    folds = []
    for j in range(CFG.n_rolling_folds):
        tr_idx, va_idx = [], []
        for key, wins in per_building.items():
            series_end = max(st for _, st in wins) + B + F
            # LINHA CRUCIAL: corte inicial (fold 0) na fração configurada,
            # alinhado ao stride; a cada fold o corte avança exatamente
            # 1 dia -> o dia recém-validado passa a integrar o treino
            # (setas da figura / modo "expanding").
            c0 = int(series_end * CFG.initial_train_frac)
            c0 -= c0 % CFG.stride
            cutoff = c0 + j * day
            for gi, st in wins:
                end = st + B + F
                if CFG.rolling_mode == "sliding":
                    in_train = end <= cutoff and st >= cutoff - CFG.train_span_steps
                else:  # expanding (figura)
                    in_train = end <= cutoff
                if in_train:
                    tr_idx.append(gi)
                # LINHA CRUCIAL: validação = janela cujo horizonte de
                # forecast é EXATAMENTE o dia seguinte ao corte:
                # backcast termina em cutoff e o alvo é [cutoff, cutoff+day).
                elif st + B >= cutoff and end <= cutoff + day:
                    va_idx.append(gi)
        if tr_idx and va_idx:
            folds.append((tr_idx, va_idx))
    if not folds:
        raise RuntimeError("Nenhum fold viável — verifique initial_train_frac "
                           "e n_rolling_folds frente ao tamanho das séries.")
    return folds


# ---------------------------------------------------------------------------
# Perdas e época (idêntico à versão anterior)
# ---------------------------------------------------------------------------
def make_criterions(cls_targets: torch.Tensor, device):
    mse = torch.nn.MSELoss()
    pos = cls_targets.sum().clamp(min=1.0)
    pw = ((cls_targets.numel() - pos) / pos).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    return mse, bce


def run_epoch(model, loader, mse, bce, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            ti = batch["trend_input"].to(device)
            si = batch["season_input"].to(device)
            tt = batch["trend_target"].to(device)
            st = batch["season_target"].to(device)
            ct = batch["cls_target"].to(device)

            t_pred, s_pred, logits = model(ti, si)
            l_t, l_s = mse(t_pred, tt), mse(s_pred, st)
            ssum = l_t + l_s                     # ponderação dinâmica original
            l_fore = (l_s / ssum) * l_t + (l_t / ssum) * l_s
            l_cls = bce(logits, ct)
            loss = l_fore + CFG.lambda_cls * l_cls

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot += loss.item() * ti.size(0)
            n += ti.size(0)
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
# Treino walk-forward (passos 6+7)
# ---------------------------------------------------------------------------
def train_model(tag="rolling", pretrained_path=None, freeze_backbone=False):
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = load_split("train")
    folds = rolling_origin_folds(full)

    model = HybridWLSTMix(device, freeze_backbone, pretrained_path).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CFG.learning_rate)

    out_dir = CFG.models_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_val_losses, history = [], []

    for j, (tr_idx, va_idx) in enumerate(folds):
        # LINHA CRUCIAL (warm start): se warm_start=True o MESMO modelo
        # continua treinando com o treino expandido — comportamento das
        # setas da figura. Se False, cada fold parte do zero (avaliação
        # rolling-origin clássica, mais cara e mais conservadora).
        if not CFG.warm_start and j > 0:
            model = HybridWLSTMix(device, freeze_backbone, pretrained_path).to(device)
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG.learning_rate)

        tr_ds, va_ds = Subset(full, tr_idx), Subset(full, va_idx)
        tl = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)
        vl = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

        # pos_weight recalculado POR FOLD, só com os rótulos do treino do
        # fold (amostrado se muito grande) — nada da validação entra aqui.
        sample = tr_idx if len(tr_idx) <= 200_000 else \
            list(np.random.default_rng(CFG.seed).choice(tr_idx, 200_000,
                                                        replace=False))
        cls_tg = torch.stack([full[i]["cls_target"] for i in sample]).flatten()
        mse, bce = make_criterions(cls_tg, device)

        best, wait = float("inf"), 0
        for ep in range(CFG.epochs_per_fold):
            tr = run_epoch(model, tl, mse, bce, device, opt)
            va = run_epoch(model, vl, mse, bce, device)
            history.append({"fold": j, "epoch": ep, "train": tr, "val": va})
            print(f"[step6-7] fold {j+1}/{len(folds)} ep {ep+1}: "
                  f"train={tr:.4f} val(dia seguinte)={va:.4f} "
                  f"| {len(tr_idx)} jan. treino / {len(va_idx)} jan. val")
            if va < best:
                best, wait = va, 0
                torch.save(model.state_dict(), out_dir / f"best_fold{j}.pth")
            else:
                wait += 1
                if wait >= CFG.patience:
                    break
        fold_val_losses.append(best)
        # Recarrega o melhor estado do fold antes de expandir o treino.
        model.load_state_dict(torch.load(out_dir / f"best_fold{j}.pth",
                                         map_location=device))

    torch.save(model.state_dict(), out_dir / "best_model.pth")
    summary = {"fold_val_losses": fold_val_losses,
               "mean": float(np.mean(fold_val_losses)),
               "std": float(np.std(fold_val_losses)),
               "mode": CFG.rolling_mode,
               "history": history}
    (out_dir / "rolling_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[step6-7] val rolling-origin: "
          f"{summary['mean']:.4f} ± {summary['std']:.4f}")
    return out_dir / "best_model.pth", summary


if __name__ == "__main__":
    train_model()
