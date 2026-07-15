"""
PASSOS 4 e 5 — Geração e fusão de rótulos de anomalia.

Passo 4 (por janela) — BICAUDAL:
  offsets = quantis 1% e 99% do resíduo (x - trend) DENTRO da janela.
  Anômalo é todo ponto FORA da banda [trend + q01, trend + q99]
  (picos acima do p99 e vales abaixo do p1).
  Como os limiares são intra-janela, nenhum percentil global vaza entre splits.

Passo 5 (fusão):
  Um mesmo timestep aparece em várias janelas (stride < janela).
  Acumulamos votos por timestep e fundimos com a regra configurada:
    - "any":      anômalo se QUALQUER janela o marcou (união);
    - "majority": anômalo se marcado em > 50% das janelas que o cobrem.
  O vetor fundido é gravado de volta no parquet do split como coluna
  `anomaly` — o dataset passa a carregar seus próprios rótulos.

`label_window` e `fuse_labels` são reutilizadas pelo passo 8 (teste
em tempo de execução), garantindo rótulos gerados pelo MESMO processo.

Visibilidade de execução:
  * Diagnóstico ANTES de processar: quantos .pt foram encontrados por
    split/setor — se estiver tudo zerado, o problema é de caminho e o
    script falha ALTO em vez de terminar em silêncio;
  * Barra tqdm por grupo: rotulagem (janelas) e fusão (prédios);
  * Log por grupo com taxa de anomalia e arquivos gravados; resumo final.
"""
from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import CFG
from step2_3_windows_wavelet import iter_windows, decompose_window

try:
    from tqdm import tqdm
except ImportError:                              # fallback sem dependência
    def tqdm(it, **kw):
        return it


def log(msg: str) -> None:
    """print imediato (flush) — evita a sensação de 'nada acontecendo'
    quando stdout está bufferizado (nohup, slurm, redirecionamento)."""
    print(msg, flush=True)


# ----------------------------- Passo 4 ------------------------------------
def label_window(window: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """Rótulos binários (uint8) de uma janela — limiar BICAUDAL.

    Anômalo todo ponto cujo resíduo (x - trend) fica ABAIXO do percentil 1
    ou ACIMA do percentil 99, ambos calculados SÓ dentro da janela.
    Geometricamente: duas curvas offsetadas, trend + q01 (banda inferior)
    e trend + q99 (banda superior); anômalo é o que sai da banda.
    """
    resid = window - trend
    # LINHAS CRUCIAIS: percentis intra-janela (sem estatística global).
    thr_low = np.quantile(resid, CFG.anomaly_quantile_low)
    thr_high = np.quantile(resid, CFG.anomaly_quantile_high)
    # x < trend + q01  OU  x > trend + q99  <=>  resid fora de [q01, q99]
    labels = (resid < thr_low) | (resid > thr_high)
    return labels.astype(np.uint8)


# ----------------------------- Passo 5 ------------------------------------
def fuse_labels(series_len: int, window_starts, window_labels) -> np.ndarray:
    """Funde rótulos de janelas sobrepostas em um vetor único por série."""
    votes = np.zeros(series_len, dtype=np.int32)     # nº de janelas que marcaram
    cover = np.zeros(series_len, dtype=np.int32)     # nº de janelas que cobrem
    W = CFG.backcast_length + CFG.forecast_length
    for st, lab in zip(window_starts, window_labels):
        votes[st: st + W] += lab
        cover[st: st + W] += 1
    if CFG.merge_rule == "any":
        fused = votes > 0                            # união (OR)
    else:                                            # majority
        with np.errstate(divide="ignore", invalid="ignore"):
            fused = np.where(cover > 0, votes / np.maximum(cover, 1) > 0.5, False)
    return fused.astype(np.uint8)


# --------------------------------------------------------------------------
# Diagnóstico prévio: o que existe para rotular?
# --------------------------------------------------------------------------
def discover(splits) -> list[tuple[str, str, Path]]:
    """Lista (split, sector, arquivo.pt) e loga o inventário.
    Se nada for encontrado, falha ALTO com o caminho esperado."""
    plan = []
    for split in splits:
        for sector in CFG.sectors:
            base = CFG.windows_root / split / sector
            if not base.exists():
                log(f"[step4-5][AVISO] não existe: {base}")
                continue
            pts = sorted(base.glob("*.pt"))
            if not pts:
                log(f"[step4-5][AVISO] sem .pt em: {base}")
            plan.extend((split, sector, pt) for pt in pts)
    log(f"[step4-5] inventário: {len(plan)} grupos (.pt), splits={tuple(splits)}")
    if not plan:
        raise RuntimeError(
            f"Nada para rotular em {CFG.windows_root}. "
            "Confira CFG.out_root/CFG.resolution e se o passo 2-3 foi executado.")
    return plan


# --------------------- Execução sobre os .pt do passo 2-3 ------------------
def label_group(split: str, sector: str, pt: Path) -> tuple[int, float]:
    pack = torch.load(pt, weights_only=False)
    x = pack["x"].numpy() if hasattr(pack["x"], "numpy") else np.asarray(pack["x"])
    trend = (pack["trend"].numpy() if hasattr(pack["trend"], "numpy")
             else np.asarray(pack["trend"]))
    N = x.shape[0]

    # Passo 4: rótulo por janela, com progresso visível.
    bar = tqdm(range(N), desc=f"rotular {split}/{sector}/{pt.stem}",
               unit="janela", file=sys.stdout, dynamic_ncols=True)
    win_labels = np.empty((N, x.shape[1]), dtype=np.uint8)
    for i in bar:
        win_labels[i] = label_window(x[i], trend[i])
    pack["labels"] = torch.tensor(win_labels)        # (N, W) por janela

    # Passo 5: fusão por prédio e escrita de volta ao parquet.
    b_idx = np.asarray(pack["building_idx"])
    starts = np.asarray(pack["start"])
    fused_labels_per_window = np.zeros_like(win_labels)
    W = CFG.backcast_length + CFG.forecast_length
    n_files = 0

    bbar = tqdm(list(enumerate(pack["buildings"])),
                desc=f"fundir  {split}/{sector}/{pt.stem}",
                unit="prédio", file=sys.stdout, dynamic_ncols=True)
    for bi, name in bbar:
        m = b_idx == bi
        if not m.any():
            continue
        fp = CFG.split_root / split / sector / pt.stem / f"{name}.parquet"
        if not fp.exists():
            log(f"[step4-5][AVISO] parquet de origem ausente, pulando: {fp}")
            continue
        df = pd.read_parquet(fp)
        fused = fuse_labels(len(df), starts[m], win_labels[m])
        df["anomaly"] = fused
        dst = CFG.labels_root / split / sector / pt.stem
        dst.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst / f"{name}.parquet", index=False)
        n_files += 1
        # LINHA CRUCIAL: reprojeta o rótulo FUNDIDO de volta em cada
        # janela — o alvo de classificação do passo 6 é o rótulo
        # consolidado, coerente entre janelas sobrepostas.
        for j in np.where(m)[0]:
            st = int(starts[j])
            fused_labels_per_window[j] = fused[st: st + W]
        if hasattr(bbar, "set_postfix"):
            bbar.set_postfix(gravados=n_files,
                             taxa=f"{fused.mean():.3%}")

    pack["labels_fused"] = torch.tensor(fused_labels_per_window)
    torch.save(pack, pt)                             # .pt atualizado in-place
    rate = float(fused_labels_per_window.mean())
    log(f"[step4-5] OK {split}/{sector}/{pt.stem}: {N} janelas rotuladas, "
        f"{n_files} parquets com coluna 'anomaly' em "
        f"{CFG.labels_root / split / sector / pt.stem} | "
        f"taxa de anomalia fundida = {rate:.4f}")
    return N, rate


def run(splits=("train",)) -> None:
    """Rotula só o split de treino — a validação rolling-origin (passo 6)
    reutiliza esses mesmos rótulos, e o teste é rotulado em runtime (passo 8)."""
    t0 = time.time()
    plan = discover(splits)
    total, rates = 0, []
    for i, (split, sector, pt) in enumerate(plan, 1):
        log(f"[step4-5] ({i}/{len(plan)}) processando {split}/{sector}/{pt.stem}...")
        n, r = label_group(split, sector, pt)
        total += n
        rates.append(r)
    log(f"[step4-5] FIM: {total} janelas rotuladas em {time.time() - t0:.1f}s "
        f"| taxa média de anomalia = {np.mean(rates):.4f} "
        f"| rótulos em {CFG.labels_root}")


if __name__ == "__main__":
    run()