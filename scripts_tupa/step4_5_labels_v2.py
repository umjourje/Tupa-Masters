"""
PASSOS 4 e 5 — Rotulagem bicaudal por janela + fusão entre janelas.

VERSÃO DUCKDB: a escrita deixa de ser 1 parquet POR PRÉDIO (milhões de
arquivos pequenos — latência de criação domina em NAS) e passa a ser UM
parquet CONSOLIDADO por shard .pt, escrito pelo DuckDB (COPY ... FORMAT
PARQUET, ZSTD), com colunas:
    building (str) | t (int64, índice temporal no split) |
    energy (float32) | anomaly (uint8)

Além disso, a série de cada prédio é RECONSTRUÍDA das próprias janelas do
.pt (a sobreposição cobre todo o trecho janelado) — nenhuma releitura dos
milhões de parquets de 01_splits é necessária.

Lógica de rotulagem/fusão INALTERADA (label_window/fuse_labels — também
usadas pelo passo 8 em runtime).

Retomada (--fine-resume implícita, sempre ativa aqui): um .pt cujo pack já
contém 'labels_fused' E cujo parquet consolidado existe é pulado.

Log: perf_log.RunLogger (terminal espelhado + detalhe por prédio +
snapshots de recursos).
"""
from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import CFG
from perf_log import RunLogger, _fmt_dur
from step2_3_windows_wavelet_v2 import iter_windows, decompose_window  # p/ passo 8

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:                                      # fallback no-op
        def __init__(self, it=None, **kw):
            self._it = it
        def __iter__(self):
            return iter(self._it or [])
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass


# ==================== rotulagem/fusão (INALTERADAS) ========================
def label_window(window: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """Bicaudal: anômalo se resíduo < q_low OU > q_high (intra-janela)."""
    resid = window - trend
    thr_low = np.quantile(resid, CFG.anomaly_quantile_low)
    thr_high = np.quantile(resid, CFG.anomaly_quantile_high)
    return ((resid < thr_low) | (resid > thr_high)).astype(np.uint8)


def fuse_labels(series_len: int, window_starts, window_labels) -> np.ndarray:
    votes = np.zeros(series_len, dtype=np.int32)
    cover = np.zeros(series_len, dtype=np.int32)
    W = CFG.backcast_length + CFG.forecast_length
    for st, lab in zip(window_starts, window_labels):
        votes[st: st + W] += lab
        cover[st: st + W] += 1
    if CFG.merge_rule == "any":
        fused = votes > 0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            fused = np.where(cover > 0,
                             votes / np.maximum(cover, 1) > 0.5, False)
    return fused.astype(np.uint8)


# ============================ escrita DuckDB ===============================
def duck_write_parquet(df: pd.DataFrame, dst: Path, logger: RunLogger) -> None:
    """LINHA CRUCIAL (escrita): UM parquet consolidado por shard via DuckDB
    (COPY vetorizado, ZSTD) — substitui milhões de escritas pequenas."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        import duckdb
        con = duckdb.connect()
        if CFG.duckdb_threads > 0:
            con.execute(f"SET threads TO {CFG.duckdb_threads}")
        con.register("df", df)
        con.execute(f"COPY df TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    except ImportError:
        logger.term("[step4-5][AVISO] duckdb indisponível; "
                    "escrevendo consolidado via pandas.")
        df.to_parquet(tmp, index=False)
    tmp.replace(dst)                                   # atômico


# ========================= processamento por .pt ===========================
def _group_dirs(split: str, pt: Path):
    rel_parent = pt.relative_to(CFG.windows_root / split).parent
    group_name = pt.stem.split(".part")[0]
    dst = (CFG.labels_root / split / rel_parent /
           f"{pt.stem}.labeled.parquet")
    return dst, group_name


def label_pack(split: str, pt: Path, logger: RunLogger,
               idx: int, total: int):
    dst, group_name = _group_dirs(split, pt)
    pack = torch.load(pt, weights_only=False)
    if "labels_fused" in pack and dst.exists():
        logger.term(f"[step4-5] ({idx}/{total}) {split}/{pt.stem}: "
                    f"pulado (já rotulado)")
        return None

    logger.group_start(f"{split}/{pt.stem}")
    t0 = time.time()
    x = np.asarray(pack["x"])
    trend = np.asarray(pack["trend"])
    b_idx = np.asarray(pack["building_idx"])
    starts = np.asarray(pack["start"])
    N, Wl = x.shape
    W = CFG.backcast_length + CFG.forecast_length

    # PASSO 4 — rótulo por janela
    t_p4 = time.time()
    win_labels = np.empty((N, Wl), dtype=np.uint8)
    for i in range(N):
        win_labels[i] = label_window(x[i], trend[i])
    pack["labels"] = torch.tensor(win_labels)
    logger.file_only(f"    passo 4 (rótulos por janela): "
                     f"{N} janelas em {_fmt_dur(time.time() - t_p4)}")

    # PASSO 5 — fusão + reconstrução da série + tabela consolidada
    t_p5 = time.time()
    fused_per_window = np.zeros_like(win_labels)
    frames = []
    bbar = tqdm(list(enumerate(pack["buildings"])),
                desc=f"({idx}/{total}) fundir {pt.stem}", unit="prédio",
                file=sys.stdout, dynamic_ncols=True)
    for bi, name in bbar:
        t_b = time.time()
        m = b_idx == bi
        if not m.any():
            continue
        st_b = starts[m]
        series_len = int(st_b.max()) + W
        # Reconstrói a série das janelas (sem reler 01_splits): a primeira
        # janela que cobre cada timestep fornece o valor.
        energy = np.full(series_len, np.nan, dtype=np.float32)
        for j_local, st in zip(np.where(m)[0], st_b):
            seg = slice(int(st), int(st) + W)
            hole = np.isnan(energy[seg])
            energy[seg][...] = np.where(hole, x[j_local], energy[seg])
        fused = fuse_labels(series_len, st_b, win_labels[m])
        for j_local, st in zip(np.where(m)[0], st_b):
            fused_per_window[j_local] = fused[int(st): int(st) + W]
        frames.append(pd.DataFrame({
            "building": name,
            "t": np.arange(series_len, dtype=np.int64),
            "energy": np.nan_to_num(energy),
            "anomaly": fused,
        }))
        logger.building(name, f"len={series_len:,} "
                        f"taxa={fused.mean():.4f} t={time.time() - t_b:.3f}s")
    bbar.close()

    table = pd.concat(frames, ignore_index=True)
    del frames
    t_w0 = time.time()
    duck_write_parquet(table, dst, logger)
    logger.file_only(f"    passo 5 (fusão): {_fmt_dur(t_w0 - t_p5)} | "
                     f"escrita duckdb: {_fmt_dur(time.time() - t_w0)} "
                     f"({len(table):,} linhas -> {dst.name}, "
                     f"{dst.stat().st_size / 1e6:.1f} MB)")

    pack["labels_fused"] = torch.tensor(fused_per_window)
    tmp = pt.with_name(pt.name + ".tmp")
    torch.save(pack, tmp)
    tmp.replace(pt)                                    # .pt atualizado, atômico
    rate = float(fused_per_window.mean())
    dt = time.time() - t0
    logger.term(f"[step4-5] ({idx}/{total}) {split}/{pt.stem}: {N} janelas "
                f"rotuladas | taxa fundida={rate:.4f} | total {_fmt_dur(dt)}")
    logger.group_end(f"{split}/{pt.stem}", f"| {_fmt_dur(dt)}")
    return N, rate


def discover(splits):
    plan = []
    for split in splits:
        base = CFG.windows_root / split
        if base.exists():
            plan.extend((split, pt) for pt in sorted(base.rglob("*.pt")))
    return plan


def run(splits=("train",), group: str | None = None) -> None:
    logger = RunLogger("step4_5")
    plan = discover(splits)
    if group:
        g = group.lower()
        plan = [(s, pt) for (s, pt) in plan
                if g in str(pt.relative_to(CFG.windows_root / s)).lower()]
        logger.term(f"[step4-5] filtro --group '{group}': "
                    f"{len(plan)} arquivo(s) .pt selecionado(s)")
    logger.term(f"[step4-5] inventário: {len(plan)} arquivos .pt, "
                f"splits={tuple(splits)}")
    if not plan:
        logger.close("vazio")
        raise RuntimeError(f"Nada para rotular em {CFG.windows_root}.")
    t0 = time.time()
    total, rates = 0, []
    for i, (split, pt) in enumerate(plan, 1):
        res = label_pack(split, pt, logger, i, len(plan))
        if res is None:
            continue
        n, r = res
        total += n
        rates.append(r)
    logger.term(f"[step4-5] FIM: {total} janelas rotuladas em "
                f"{_fmt_dur(time.time() - t0)} | taxa média = "
                f"{(np.mean(rates) if rates else float('nan')):.4f} | "
                f"rótulos em {CFG.labels_root}")
    logger.close(f"janelas={total}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, default=None,
                    help="rotula só os .pt cujo caminho contém este texto "
                         "(ex.: 'HRSA/11', 'state=AL.part000', 'SynD')")
    run(group=ap.parse_args().group)