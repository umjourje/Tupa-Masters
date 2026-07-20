"""
PASSOS 2 e 3 — Janelamento + decomposição wavelet por janela -> shards .pt.

VERSÃO VETORIZADA + PARALELA (para os 512 GB / muitos núcleos):
  * VETORIZAÇÃO: por prédio, TODAS as janelas são construídas de uma vez
    (numpy sliding_window_view) e a wavelet roda UMA única chamada em C
    (pywt.wavedec/waverec com axis=-1) sobre a matriz N_janelas × W —
    substitui o loop Python janela-a-janela (765M iterações no HRSA/01,
    causa dos 2+ dias travado em 1 núcleo).
  * PARALELISMO: os prédios do grupo são divididos em chunks processados
    por um pool de PROCESSOS (config.workers); cada worker lê seu chunk
    via DuckDB, processa vetorizado e grava seus PRÓPRIOS shards
    (<grupo>.wNN.partKKK.pt) e manifesto (<grupo>.wNN.shards.jsonl).
  * LOG EM BLOCOS: cada worker escreve as linhas por-prédio de um shard
    de UMA vez em seu arquivo de detalhe (logs/..._wNN.detail.log) —
    sem milhões de flushes no NAS.
  * Retomada: .done por grupo + fine-resume via união de TODOS os
    manifestos *.shards.jsonl (prédios concluídos nem são lidos).
  * Graceful stop: Ctrl+C cancela chunks pendentes; chunks concluídos
    ficam retomáveis.
"""
from __future__ import annotations
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import torch
from numpy.lib.stride_tricks import sliding_window_view

from config import CFG
from perf_log import RunLogger, _fmt_dur

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, it=None, **kw): self._it = it
        def __iter__(self): return iter(self._it or [])
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass

_BATCH_HARD_CAP = 65536


# ================= núcleo por-janela (API mantida p/ passo 8) =============
def decompose_window(window: np.ndarray, wavelet: str = CFG.wavelet,
                     level: int = CFG.wavelet_level):
    t, s = decompose_windows_batch(window[None, :], wavelet, level)
    return t[0], s[0]


def decompose_windows_batch(wins: np.ndarray, wavelet: str = CFG.wavelet,
                            level: int = CFG.wavelet_level):
    """LINHA CRUCIAL (vetorização): decompõe TODAS as janelas (N, W) numa
    única chamada C do pywt (axis=-1) — o loop Python por janela some."""
    W = wins.shape[-1]
    max_lv = pywt.dwt_max_level(W, pywt.Wavelet(wavelet).dec_len)
    lv = min(level, max_lv)
    coeffs = pywt.wavedec(wins, wavelet, level=lv, axis=-1)
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet, axis=-1)[..., :W]
    season = wins - trend
    return trend.astype(np.float32), np.nan_to_num(season).astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-8):
    m, s = float(np.mean(x)), float(np.std(x))
    return ((x - m) / (s + eps)).astype(np.float32), m, s


def standardize_batch(x: np.ndarray, eps: float = 1e-8):
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32), m.ravel(), s.ravel()


def iter_windows(series: np.ndarray):
    W = CFG.backcast_length + CFG.forecast_length
    for start in range(0, len(series) - W + 1, CFG.stride):
        yield start, series[start: start + W]


def make_windows(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Janelamento vetorizado: (starts, matriz N×W) sem loop Python."""
    W = CFG.backcast_length + CFG.forecast_length
    if len(series) < W:
        return np.empty(0, np.int64), np.empty((0, W))
    wins = sliding_window_view(series, W)[::CFG.stride]
    starts = np.arange(wins.shape[0], dtype=np.int64) * CFG.stride
    return starts, np.ascontiguousarray(wins)


def n_windows(series_len: int) -> int:
    W = CFG.backcast_length + CFG.forecast_length
    return 0 if series_len < W else (series_len - W) // CFG.stride + 1


# ============================ DuckDB (por worker) ==========================
def _duck_con():
    import duckdb
    con = duckdb.connect()
    if CFG.duckdb_threads > 0:
        con.execute(f"SET threads TO {CFG.duckdb_threads}")
    con.execute(f"SET memory_limit='{CFG.duckdb_memory_limit_gb}GB'")
    return con


def _read_files(con, paths: list[str]) -> pd.DataFrame:
    if con is None:                              # fallback sem duckdb
        frames = []
        for p in paths:
            d = pd.read_parquet(p)[["energy"]]
            d["filename"] = p
            d["rn"] = np.arange(len(d))
            frames.append(d)
        return pd.concat(frames, ignore_index=True)
    lst = "[" + ",".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"
    return con.execute(
        f"SELECT filename, file_row_number AS rn, energy "
        f"FROM read_parquet({lst}, filename=true, file_row_number=true)").df()


# ============================ worker (processo) ============================
def _worker(args) -> dict:
    (chunk_paths, split, group_rel_s, group_name, wid, detail_log) = args
    group_rel = Path(group_rel_s)
    try:
        con = _duck_con()
    except Exception:
        con = None
    out_dir = CFG.windows_root / split / group_rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = len(list(out_dir.glob(f"{group_name}.w{wid:02d}.part*.pt")))
    manifest = out_dir / f"{group_name}.w{wid:02d}.shards.jsonl"

    buf = {k: [] for k in ("x", "trend", "season", "trend_norm",
                           "season_norm", "building_idx", "start", "stats")}
    buildings, block_lines = [], []
    total_w = 0
    t0 = time.time()

    def flush():
        nonlocal seq, total_w, buf, buildings, block_lines
        if not buf["x"]:
            return
        out = {
            "x": torch.tensor(np.concatenate(buf["x"]).astype(np.float32)),
            "trend": torch.tensor(np.concatenate(buf["trend"])),
            "season": torch.tensor(np.concatenate(buf["season"])),
            "trend_norm": torch.tensor(np.concatenate(buf["trend_norm"])),
            "season_norm": torch.tensor(np.concatenate(buf["season_norm"])),
            "building_idx": torch.tensor(np.concatenate(buf["building_idx"]),
                                         dtype=torch.long),
            "start": torch.tensor(np.concatenate(buf["start"]),
                                  dtype=torch.long),
            "stats": torch.tensor(np.concatenate(buf["stats"])),
            "buildings": buildings,
            "meta": {"backcast": CFG.backcast_length,
                     "forecast": CFG.forecast_length, "stride": CFG.stride,
                     "wavelet": CFG.wavelet, "level": CFG.wavelet_level,
                     "shard": seq, "worker": wid},
        }
        name = f"{group_name}.w{wid:02d}.part{seq:03d}.pt"
        tmp = out_dir / (name + ".tmp")
        torch.save(out, tmp)
        tmp.replace(out_dir / name)
        with open(manifest, "a") as f:
            f.write(json.dumps({"shard": seq, "file": name,
                                "buildings": buildings}) + "\n")
        # LOG EM BLOCO: todas as linhas por-prédio do shard, de uma vez.
        with open(detail_log, "a") as f:
            f.write(f"--- shard {name} "
                    f"({int(out['x'].shape[0])} janelas) ---\n")
            f.write("\n".join(block_lines) + "\n")
        total_w += int(out["x"].shape[0])
        seq += 1
        buf = {k: [] for k in buf}
        buildings, block_lines = [], []

    B = min(CFG.duckdb_files_per_batch, _BATCH_HARD_CAP)
    buffered = 0
    for k in range(0, len(chunk_paths), B):
        df = _read_files(con, chunk_paths[k:k + B])
        for name, g in df.groupby("filename", sort=True):
            s = np.nan_to_num(g.sort_values("rn")["energy"]
                              .to_numpy(np.float64))
            starts, wins = make_windows(s)
            if wins.shape[0] == 0:
                continue
            t_b = time.time()
            trend, season = decompose_windows_batch(wins)   # 1 chamada C
            t_n, tm, ts = standardize_batch(trend)
            s_n, sm, ss = standardize_batch(season)
            b_local = len(buildings)
            buildings.append(Path(name).stem)
            nb = wins.shape[0]
            buf["x"].append(wins.astype(np.float32))
            buf["trend"].append(trend)
            buf["season"].append(season)
            buf["trend_norm"].append(t_n)
            buf["season_norm"].append(s_n)
            buf["building_idx"].append(np.full(nb, b_local, np.int64))
            buf["start"].append(starts)
            buf["stats"].append(np.stack([tm, ts, sm, ss], axis=1))
            block_lines.append(f"      {Path(name).stem}: janelas={nb} "
                               f"linhas={len(s):,} t={time.time()-t_b:.4f}s")
            buffered += nb
            if buffered >= CFG.max_windows_per_shard:
                flush()
                buffered = 0
        del df
    flush()
    return {"worker": wid, "windows": total_w,
            "buildings": len(chunk_paths), "secs": time.time() - t0}


# ====================== descoberta / orquestração ==========================
def _find_leaf_dirs(base: Path) -> list[Path]:
    leaves = []
    def rec(d: Path):
        subdirs = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".parquet"):
                        leaves.append(d)
                        return
                    if e.is_dir():
                        subdirs.append(Path(e.path))
        except OSError:
            return
        for s in sorted(subdirs):
            rec(s)
    rec(base)
    return sorted(leaves)


def discover(group: str | None = None):
    plan = []
    for split in CFG.splits:
        base = CFG.split_root / split
        if not base.exists():
            continue
        for gdir in _find_leaf_dirs(base):
            rel = str(gdir.relative_to(base)).lower()
            if group and group.lower() not in rel:
                continue
            plan.append((split, gdir))
    return plan


def _marker_path23(split: str, group_rel: Path) -> Path:
    safe = f"{split}__" + str(group_rel).replace("\\", "__").replace("/", "__")
    return CFG.windows_root / "_manifest_step23" / f"{safe}.done"


def build_group(split, gdir, logger, fine_resume, idx, total_groups) -> int:
    group_rel = gdir.relative_to(CFG.split_root / split)
    group_name = group_rel.name if str(group_rel) != "." else "root"
    marker = _marker_path23(split, group_rel)
    if marker.exists():
        logger.term(f"[step2-3] ({idx}/{total_groups}) {split}/{group_rel}: "
                    f"pulado (já concluído)")
        return 0

    out_dir = CFG.windows_root / split / group_rel.parent
    t_ls = time.time()
    files = sorted(f for f in gdir.glob("*.parquet") if f.is_file())
    logger.file_only(f"    listagem: {len(files):,} arquivos em "
                     f"{_fmt_dur(time.time() - t_ls)}")

    manifests = sorted(out_dir.glob(f"{group_name}.w*.shards.jsonl")) + \
        ([out_dir / f"{group_name}.shards.jsonl"]
         if (out_dir / f"{group_name}.shards.jsonl").exists() else [])
    if fine_resume and manifests:
        done = set()
        for mf in manifests:
            for ln in mf.read_text().splitlines():
                done.update(json.loads(ln)["buildings"])
        files = [f for f in files if f.stem not in done]
        logger.term(f"[step2-3][fine-resume] {split}/{group_rel}: "
                    f"{len(done)} prédios já concluídos pulados; "
                    f"restam {len(files)}")
    elif not fine_resume:
        for stale in (list(out_dir.glob(f"{group_name}.pt")) +
                      list(out_dir.glob(f"{group_name}.part*.pt")) +
                      list(out_dir.glob(f"{group_name}.w*.pt")) +
                      list(out_dir.glob(f"{group_name}*.tmp")) + manifests):
            stale.unlink()

    if not files:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"windows": 0, "note": "nada a fazer"}))
        return 0

    n_workers = CFG.workers or max(1, (os.cpu_count() or 4) - 2)
    chunk = min(CFG.buildings_per_chunk,
                max(64, len(files) // (n_workers * 4) + 1))
    chunks = [files[i:i + chunk] for i in range(0, len(files), chunk)]
    logger.group_start(f"{split}/{group_rel} — {len(files):,} prédios, "
                       f"{len(chunks)} chunks × ~{chunk}, "
                       f"{n_workers} workers")
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    detail_base = CFG.out_root / "logs"

    t0 = time.time()
    total_w = 0
    bar = tqdm(total=len(files), desc=f"({idx}/{total_groups}) "
               f"{split}/{group_rel}", unit="prédio",
               file=sys.stdout, dynamic_ncols=True)
    tasks = [([str(f) for f in c], split, str(group_rel), group_name,
              w % n_workers,
              str(detail_base / f"step2_3_{ts_tag}_w{w % n_workers:02d}.detail.log"))
             for w, c in enumerate(chunks)]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                total_w += r["windows"]
                bar.update(r["buildings"])
                bar.set_postfix(janelas=total_w)
                logger.file_only(f"    chunk ok (w{r['worker']:02d}): "
                                 f"{r['buildings']} prédios, "
                                 f"{r['windows']} janelas, "
                                 f"{_fmt_dur(r['secs'])}")
                logger.snapshot("pós-chunk")
    except KeyboardInterrupt:
        bar.close()
        logger.term(f"[step2-3] PARADA: chunks pendentes cancelados; "
                    f"{total_w} janelas gravadas até aqui — retome com "
                    f"--fine-resume.")
        logger.group_end(f"{split}/{group_rel}", "| INTERROMPIDO")
        raise SystemExit(130)
    bar.close()

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"windows": total_w}))
    dt = time.time() - t0
    logger.term(f"[step2-3] ({idx}/{total_groups}) {split}/{group_rel}: "
                f"{total_w} janelas | {n_workers} workers | "
                f"total {_fmt_dur(dt)}")
    logger.group_end(f"{split}/{group_rel}", f"| {_fmt_dur(dt)}")
    return total_w


def run(fine_resume: bool = False, group: str | None = None) -> None:
    logger = RunLogger("step2_3")
    if CFG.duckdb_files_per_batch > _BATCH_HARD_CAP:
        logger.term(f"[step2-3][AVISO] duckdb_files_per_batch absurdo; "
                    f"usando teto {_BATCH_HARD_CAP}.")
    t_d = time.time()
    plan = discover(group)
    logger.term(f"[step2-3] inventário: {len(plan)} grupo(s) "
                f"{'(filtro: ' + group + ') ' if group else ''}"
                f"em {_fmt_dur(time.time() - t_d)}, splits={CFG.splits}")
    if not plan:
        logger.close("vazio")
        raise RuntimeError(f"Nada para processar em {CFG.split_root}.")
    t0 = time.time()
    grand = 0
    for i, (split, gdir) in enumerate(plan, 1):
        grand += build_group(split, gdir, logger, fine_resume, i, len(plan))
    logger.term(f"[step2-3] FIM: {grand} janelas em "
                f"{_fmt_dur(time.time() - t0)} | saída em {CFG.windows_root}")
    logger.close(f"janelas={grand}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine-resume", action="store_true")
    ap.add_argument("--group", type=str, default=None)
    a = ap.parse_args()
    run(fine_resume=a.fine_resume, group=a.group)