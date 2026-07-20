"""
PASSO 6 — Treino do modelo híbrido (backbone W-LSTMix + bloco de
classificação ao final) e teste com rotulagem em tempo real.

MODOS (flag --mode):
  train  : lê os shards .pt já prontos (02_windows, com labels_fused do
           passo 4-5) e treina com validação rolling-origin. NENHUMA
           rotulagem acontece aqui — os dados já estão prontos em disco.
  test   : recebe dados CRUS no formato do dataset puro (timestamp +
           medida), e SÓ AQUI os passos 3-5 rodam em tempo de execução
           (funções vetorizadas importadas dos passos 2-3/4-5) para gerar
           os rótulos dos dados novos e medir o modelo.

Arquitetura: models/W_LSTMix.py ORIGINAL, inalterado; a única adição é a
cabeça de classificação (model_hybrid.HybridWLSTMix) — um logit por
timestep do horizonte, treinada com perda conjunta forecast + BCE.

Velocidade (máquina grande):
  * TF32 + cudnn.benchmark;
  * AMP (mixed precision) com GradScaler (CFG.use_amp);
  * DataLoader com CFG.loader_workers, pin_memory, persistent_workers e
    prefetch — a GPU não espera o disco.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from config import CFG
from perf_log import RunLogger, _fmt_dur
from model_hybrid import HybridWLSTMix
from step2_3_windows_wavelet import (make_windows, decompose_windows_batch,
                                     standardize_batch)
from step4_5_labels import label_windows_batch, fuse_labels


def _speed_setup():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ============================== TREINO =====================================
class WindowedPTDataset(Dataset):
    def __init__(self, pt_path):
        pack = torch.load(pt_path, weights_only=False)
        B = CFG.backcast_length
        self.trend_in = pack["trend_norm"][:, :B]
        self.season_in = pack["season_norm"][:, :B]
        self.trend_tg = pack["trend_norm"][:, B:]
        self.season_tg = pack["season_norm"][:, B:]
        self.cls_tg = pack["labels_fused"][:, B:].float()
        self.building_idx = pack["building_idx"]
        self.starts = pack["start"]
        self.group = pt_path.stem                 # inclui .wNN.partKKK: único

    def __len__(self):
        return self.trend_in.shape[0]

    def __getitem__(self, i):
        return {"trend_input": self.trend_in[i],
                "season_input": self.season_in[i],
                "trend_target": self.trend_tg[i],
                "season_target": self.season_tg[i],
                "cls_target": self.cls_tg[i]}


def load_split(split: str) -> ConcatDataset:
    base = CFG.windows_root / split
    ds = [WindowedPTDataset(p) for p in sorted(base.rglob("*.pt"))] \
        if base.exists() else []
    if not ds:
        raise RuntimeError(f"Nenhum .pt em {base} — rode os passos 1-5.")
    return ConcatDataset(ds)


def rolling_origin_folds(full: ConcatDataset):
    from collections import defaultdict
    B, F = CFG.backcast_length, CFG.forecast_length
    day = CFG.val_horizon_steps
    per_building = defaultdict(list)
    offset = 0
    for ds in full.datasets:
        for i in range(len(ds)):
            key = f"{ds.group}:{int(ds.building_idx[i])}"
            per_building[key].append((offset + i, int(ds.starts[i])))
        offset += len(ds)
    folds = []
    for j in range(CFG.n_rolling_folds):
        tr, va = [], []
        for key, wins in per_building.items():
            series_end = max(st for _, st in wins) + B + F
            c0 = int(series_end * CFG.initial_train_frac)
            c0 -= c0 % CFG.stride
            cutoff = c0 + j * day
            for gi, st in wins:
                end = st + B + F
                if CFG.rolling_mode == "sliding":
                    in_train = end <= cutoff and st >= cutoff - CFG.train_span_steps
                else:
                    in_train = end <= cutoff
                if in_train:
                    tr.append(gi)
                elif st + B >= cutoff and end <= cutoff + day:
                    va.append(gi)
        if tr and va:
            folds.append((tr, va))
    if not folds:
        raise RuntimeError("Nenhum fold viável — ajuste initial_train_frac.")
    return folds


def _loader(ds, shuffle):
    return DataLoader(ds, batch_size=CFG.batch_size, shuffle=shuffle,
                      num_workers=CFG.loader_workers, pin_memory=True,
                      persistent_workers=CFG.loader_workers > 0,
                      prefetch_factor=4 if CFG.loader_workers > 0 else None)


def run_epoch(model, loader, mse, bce, device, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot, n = 0.0, 0
    amp = CFG.use_amp and device.type == "cuda"
    with torch.set_grad_enabled(training):
        for batch in loader:
            ti = batch["trend_input"].to(device, non_blocking=True)
            si = batch["season_input"].to(device, non_blocking=True)
            tt = batch["trend_target"].to(device, non_blocking=True)
            st = batch["season_target"].to(device, non_blocking=True)
            ct = batch["cls_target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                t_pred, s_pred, logits = model(ti, si)
                l_t, l_s = mse(t_pred, tt), mse(s_pred, st)
                ssum = l_t + l_s
                l_fore = (l_s / ssum) * l_t + (l_t / ssum) * l_s
                loss = l_fore + CFG.lambda_cls * bce(logits, ct)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            tot += loss.item() * ti.size(0)
            n += ti.size(0)
    return tot / max(n, 1)


def train(pretrained_path=None, freeze_backbone=False, tag="rolling"):
    logger = RunLogger("step6_train")
    _speed_setup()
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.term(f"[step6] device={device} amp={CFG.use_amp} "
                f"batch={CFG.batch_size} loader_workers={CFG.loader_workers}")
    full = load_split("train")
    logger.term(f"[step6] {sum(len(d) for d in full.datasets):,} janelas "
                f"em {len(full.datasets)} shards")
    folds = rolling_origin_folds(full)

    model = HybridWLSTMix(device, freeze_backbone, pretrained_path).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=CFG.learning_rate)
    scaler = torch.amp.GradScaler(enabled=CFG.use_amp and
                                  device.type == "cuda")
    out_dir = CFG.models_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_losses, history = [], []

    for j, (tr_idx, va_idx) in enumerate(folds):
        if not CFG.warm_start and j > 0:
            model = HybridWLSTMix(device, freeze_backbone,
                                  pretrained_path).to(device)
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                          model.parameters()),
                                   lr=CFG.learning_rate)
        tl = _loader(Subset(full, tr_idx), True)
        vl = _loader(Subset(full, va_idx), False)
        sample = tr_idx[:200_000]
        ys = torch.stack([full[i]["cls_target"] for i in sample]).flatten()
        pos = ys.sum().clamp(min=1.0)
        bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=((ys.numel() - pos) / pos).to(device))
        mse = torch.nn.MSELoss()

        best, wait = float("inf"), 0
        for ep in range(CFG.epochs_per_fold):
            t_e = time.time()
            tr = run_epoch(model, tl, mse, bce, device, scaler, opt)
            va = run_epoch(model, vl, mse, bce, device, scaler)
            history.append({"fold": j, "epoch": ep, "train": tr, "val": va})
            logger.term(f"[step6] fold {j+1}/{len(folds)} ep {ep+1}: "
                        f"train={tr:.4f} val={va:.4f} "
                        f"({_fmt_dur(time.time() - t_e)})")
            logger.snapshot(f"fold{j}_ep{ep}")
            if va < best:
                best, wait = va, 0
                torch.save(model.state_dict(), out_dir / f"best_fold{j}.pth")
            else:
                wait += 1
                if wait >= CFG.patience:
                    break
        fold_losses.append(best)
        model.load_state_dict(torch.load(out_dir / f"best_fold{j}.pth",
                                         map_location=device))

    torch.save(model.state_dict(), out_dir / "best_model.pth")
    (out_dir / "rolling_results.json").write_text(json.dumps(
        {"fold_val_losses": fold_losses,
         "mean": float(np.mean(fold_losses)),
         "std": float(np.std(fold_losses)), "history": history}, indent=2))
    logger.term(f"[step6] FIM treino: val {np.mean(fold_losses):.4f} "
                f"± {np.std(fold_losses):.4f} | modelo em "
                f"{out_dir / 'best_model.pth'}")
    logger.close("train")


# =============================== TESTE =====================================
def _load_raw_series(path: Path):
    """Dados novos no formato do dataset puro (timestamp + medida).
    Usa o extrator do passo 1 quando disponível (wide/long/single)."""
    try:
        from step1_split import iter_building_series_file
        for bname, bdf in iter_building_series_file(path):
            if "timestamp" in bdf.columns:
                bdf = bdf.sort_values("timestamp")
            yield bname, np.nan_to_num(bdf["energy"].to_numpy(np.float64))
        return
    except ImportError:
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path))
        tcol = next((c for c in df.columns
                     if c.lower() in ("timestamp", "datetime", "date")), None)
        if tcol:
            df = df.sort_values(tcol)
        vcol = next(c for c in df.columns if c != tcol)
        yield path.stem, np.nan_to_num(df[vcol].to_numpy(np.float64))


@torch.no_grad()
def test(data: Path, model_path=None, threshold: float = 0.5):
    """Rotulagem EM TEMPO REAL (passos 3-5 vetorizados) só aqui, para
    dados novos, seguida da avaliação do modelo."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    logger = RunLogger("step6_test")
    _speed_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridWLSTMix(device).to(device)
    model_path = model_path or (CFG.models_root / "rolling" / "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.term(f"[step6:test] modelo={model_path} device={device}")

    B, F = CFG.backcast_length, CFG.forecast_length
    files = ([data] if data.is_file()
             else sorted(p for p in data.rglob("*")
                         if p.suffix in (".parquet", ".csv")))
    Y, P = [], []
    for fp in files:
        for bname, s in _load_raw_series(fp):
            starts, wins = make_windows(s)
            if wins.shape[0] == 0:
                continue
            # PASSOS 3-5 EM RUNTIME (vetorizados, mesmas funções do treino):
            trend, season = decompose_windows_batch(wins)
            win_labels = label_windows_batch(wins, trend)
            y_true = fuse_labels(len(s), starts, win_labels)
            t_n, *_ = standardize_batch(trend)
            s_n, *_ = standardize_batch(season)
            ti = torch.tensor(t_n[:, :B]).to(device)
            si = torch.tensor(s_n[:, :B]).to(device)
            probs = []
            for i in range(0, len(ti), CFG.batch_size):
                with torch.autocast(device_type=device.type,
                                    enabled=CFG.use_amp and
                                    device.type == "cuda"):
                    _, _, logits = model(ti[i:i+CFG.batch_size],
                                         si[i:i+CFG.batch_size])
                probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            probs = np.concatenate(probs)
            prob_sum = np.zeros(len(s)); prob_cnt = np.zeros(len(s))
            for j, st in enumerate(starts):
                prob_sum[st+B: st+B+F] += probs[j]
                prob_cnt[st+B: st+B+F] += 1
            cov = prob_cnt > 0
            Y.append(y_true[cov])
            P.append(prob_sum[cov] / prob_cnt[cov])
            logger.building(f"{fp.stem}:{bname}",
                            f"janelas={wins.shape[0]} "
                            f"taxa_rotulada={y_true.mean():.4f}")
    y = np.concatenate(Y); p = np.concatenate(P)
    pred = (p >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": (float(roc_auc_score(y, p))
                  if y.min() != y.max() else None),
        "anomaly_rate_true": float(y.mean()), "n_points": int(len(y)),
    }
    (CFG.models_root / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    logger.term(f"[step6:test] {json.dumps(metrics)}")
    logger.close("test")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--data", type=Path, default=None,
                    help="(test) arquivo ou pasta com dados CRUS "
                         "(timestamp + medida)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="(train) pesos do W-LSTMix p/ fine-tuning")
    ap.add_argument("--freeze-backbone", action="store_true")
    a = ap.parse_args()
    if a.mode == "train":
        train(pretrained_path=a.pretrained,
              freeze_backbone=a.freeze_backbone)
    else:
        if a.data is None:
            ap.error("--mode test exige --data")
        test(a.data, model_path=a.model)