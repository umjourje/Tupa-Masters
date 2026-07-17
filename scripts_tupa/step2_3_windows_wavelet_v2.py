"""
PASSOS 2 e 3 — Janelamento (overlapping sliding window) por grupo e
decomposição wavelet POR JANELA, salva em .pt (shards) para o PyTorch.

VERSÃO DUCKDB: a leitura dos milhões de parquets pequenos de 01_splits
deixa de ser 1 pd.read_parquet POR ARQUIVO (latência de abertura domina;
foi o gargalo das 11h do passo 1) e passa a ser UMA query DuckDB por LOTE
de arquivos:
    SELECT filename, file_row_number, energy
    FROM read_parquet([...lote...], filename=true, file_row_number=true)
— varredura vetorizada e paralela (usa todos os núcleos), com a ordem
temporal de cada série garantida por file_row_number.

O Python/numpy segue responsável apenas pelo que é por-janela:
iter_windows + decompose_window (wavelet) + normalização — inalterados.

Retomada:
  * marcador de GRUPO (.done) como antes;
  * --fine-resume DENTRO do grupo: um manifesto .shards.jsonl registra os
    prédios de cada shard CONCLUÍDO; na retomada, esses prédios são
    removidos da lista de arquivos ANTES da leitura (nem são lidos) e a
    numeração de shards continua de onde parou.

Log: perf_log.RunLogger — terminal minimalista espelhado + detalhes por
edifício indentados por lote/grupo + snapshots de recursos.
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np
import pandas as pd
import pywt
import torch
from pathlib import Path
from config import CFG
from perf_log import RunLogger, _fmt_dur

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


# ======================= núcleo por-janela (INALTERADO) ====================
def decompose_window(window: np.ndarray,
                     wavelet: str = CFG.wavelet,
                     level: int = CFG.wavelet_level):
    """Decompõe UMA janela em (trend, seasonal+residual) via DWT."""
    max_lv = pywt.dwt_max_level(len(window), pywt.Wavelet(wavelet).dec_len)
    lv = min(level, max_lv)
    coeffs = pywt.wavedec(window, wavelet, level=lv)
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet)[: len(window)]
    season = window - trend
    return trend.astype(np.float32), np.nan_to_num(season).astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-8):
    m, s = float(np.mean(x)), float(np.std(x))
    return ((x - m) / (s + eps)).astype(np.float32), m, s


def iter_windows(series: np.ndarray):
    W = CFG.backcast_length + CFG.forecast_length
    for start in range(0, len(series) - W + 1, CFG.stride):
        yield start, series[start: start + W]


def n_windows(series_len: int) -> int:
    W = CFG.backcast_length + CFG.forecast_length
    return 0 if series_len < W else (series_len - W) // CFG.stride + 1


# ============================ leitura DuckDB ===============================
def _duck_con():
    import duckdb
    con = duckdb.connect()
    if CFG.duckdb_threads > 0:
        con.execute(f"SET threads TO {CFG.duckdb_threads}")
    con.execute(f"SET memory_limit='{CFG.duckdb_memory_limit_gb}GB'")
    return con


def duck_read_batch(con, paths: list[str]) -> pd.DataFrame:
    """LINHA CRUCIAL (leitura): um lote inteiro de parquets em UMA query;
    file_row_number preserva a ordem temporal dentro de cada arquivo."""
    lst = "[" + ",".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"
    return con.execute(
        f"SELECT filename, file_row_number AS rn, energy "
        f"FROM read_parquet({lst}, filename=true, file_row_number=true)"
    ).df()


def iter_group_series(files: list[Path], logger: RunLogger, bar):
    """Itera (nome_do_prédio, série np) do grupo, lote a lote via DuckDB.
    Fallback pandas por arquivo se o duckdb não estiver disponível."""
    try:
        con = _duck_con()
    except Exception as e:
        logger.term(f"[step2-3][AVISO] duckdb indisponível ({e}); "
                    f"fallback pandas por arquivo (LENTO).")
        for fp in files:
            s = np.nan_to_num(pd.read_parquet(fp)["energy"]
                              .to_numpy(np.float64))
            bar.update(1)
            yield fp.stem, s
        return

    B = CFG.duckdb_files_per_batch
    for k in range(0, len(files), B):
        batch = files[k:k + B]
        t0 = time.time()
        df = duck_read_batch(con, [str(f) for f in batch])
        t_read = time.time() - t0
        logger.file_start(f"lote {k // B + 1} ({len(batch)} arquivos)")
        logger.file_only(f"    leitura duckdb: {_fmt_dur(t_read)} "
                         f"({len(df):,} linhas)")
        for name, g in df.groupby("filename", sort=True):
            s = np.nan_to_num(g.sort_values("rn")["energy"]
                              .to_numpy(np.float64))
            bar.update(1)
            yield Path(name).stem, s
        logger.file_end(f"lote {k // B + 1}")
        del df


# =========================== shards + retomada =============================
def _empty_buf():
    return {k: [] for k in ("x", "trend", "season", "trend_norm",
                            "season_norm", "building_idx", "start", "stats")}


def _marker_path23(split: str, group_rel: Path) -> Path:
    safe = f"{split}__" + str(group_rel).replace("\\", "__").replace("/", "__")
    return CFG.windows_root / "_manifest_step23" / f"{safe}.done"


def _shards_manifest(split: str, group_rel: Path, group_name: str) -> Path:
    return (CFG.windows_root / split / group_rel.parent /
            f"{group_name}.shards.jsonl")


def _flush_shard(buf, buildings, split, group_rel, group_name,
                 shard_idx, multi, logger) -> int:
    if not buf["x"]:
        return 0
    out = {
        "x": torch.tensor(np.stack(buf["x"])),
        "trend": torch.tensor(np.stack(buf["trend"])),
        "season": torch.tensor(np.stack(buf["season"])),
        "trend_norm": torch.tensor(np.stack(buf["trend_norm"])),
        "season_norm": torch.tensor(np.stack(buf["season_norm"])),
        "building_idx": torch.tensor(buf["building_idx"], dtype=torch.long),
        "start": torch.tensor(buf["start"], dtype=torch.long),
        "stats": torch.tensor(buf["stats"]),
        "buildings": buildings,
        "meta": {"backcast": CFG.backcast_length,
                 "forecast": CFG.forecast_length, "stride": CFG.stride,
                 "wavelet": CFG.wavelet, "level": CFG.wavelet_level,
                 "shard": shard_idx},
    }
    name = f"{group_name}.part{shard_idx:03d}.pt" if multi else f"{group_name}.pt"
    dst = CFG.windows_root / split / group_rel.parent / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(out, tmp)
    tmp.replace(dst)                                   # escrita atômica
    # LINHA CRUCIAL (fine-resume): o manifesto registra os prédios do shard
    # APÓS a gravação atômica — na retomada, esses prédios nem são lidos.
    with open(_shards_manifest(split, group_rel, group_name), "a") as f:
        f.write(json.dumps({"shard": shard_idx, "file": name,
                            "buildings": buildings}) + "\n")
    mb = dst.stat().st_size / 1e6
    logger.file_only(f"    shard gravado: {name} "
                     f"({out['x'].shape[0]} janelas, {len(buildings)} prédios, "
                     f"{mb:.1f} MB)")
    return int(out["x"].shape[0])


def build_group(split: str, gdir: Path, files: list[Path],
                logger: RunLogger, fine_resume: bool,
                idx: int, total_groups: int) -> int:
    group_rel = gdir.relative_to(CFG.split_root / split)
    group_name = group_rel.name if str(group_rel) != "." else "root"
    marker = _marker_path23(split, group_rel)
    if marker.exists():
        logger.term(f"[step2-3] ({idx}/{total_groups}) {split}/{group_rel}: "
                    f"pulado (já concluído)")
        return 0

    out_dir = CFG.windows_root / split / group_rel.parent
    manifest = _shards_manifest(split, group_rel, group_name)
    done_buildings: set = set()
    shard_idx = 0
    if fine_resume and manifest.exists():
        for ln in manifest.read_text().splitlines():
            rec = json.loads(ln)
            done_buildings.update(rec["buildings"])
            shard_idx = max(shard_idx, rec["shard"] + 1)
        files = [f for f in files if f.stem not in done_buildings]
        logger.term(f"[step2-3][fine-resume] {split}/{group_rel}: "
                    f"{len(done_buildings)} prédios em {shard_idx} shards "
                    f"concluídos serão pulados")
    else:
        if out_dir.exists():                          # refaz do zero
            for stale in (list(out_dir.glob(f"{group_name}.pt")) +
                          list(out_dir.glob(f"{group_name}.part*.pt")) +
                          list(out_dir.glob(f"{group_name}*.tmp"))):
                stale.unlink()
        if manifest.exists():
            manifest.unlink()

    multi = fine_resume and shard_idx > 0
    logger.group_start(f"{split}/{group_rel} ({len(files)} prédios a processar)")
    t_g0 = time.time()
    buf, buildings = _empty_buf(), []
    total_w, shards_written = 0, 0
    bar = tqdm(total=len(files), desc=f"({idx}/{total_groups}) "
               f"{split}/{group_rel}", unit="prédio",
               file=sys.stdout, dynamic_ncols=True)

    for bname, s in iter_group_series(files, logger, bar):
        t_b0 = time.time()
        b_local = len(buildings)
        buildings.append(bname)
        nw = 0
        for start, w in iter_windows(s):
            trend, season = decompose_window(w)
            t_n, tm, ts = standardize(trend)
            s_n, sm, ss = standardize(season)
            buf["x"].append(w.astype(np.float32))
            buf["trend"].append(trend)
            buf["season"].append(season)
            buf["trend_norm"].append(t_n)
            buf["season_norm"].append(s_n)
            buf["building_idx"].append(b_local)
            buf["start"].append(start)
            buf["stats"].append([tm, ts, sm, ss])
            nw += 1
        logger.building(bname, f"janelas={nw} linhas={len(s):,} "
                        f"t={time.time() - t_b0:.3f}s")
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(janelas=total_w + len(buf["x"]), shard=shard_idx)
        if len(buf["x"]) >= CFG.max_windows_per_shard:
            total_w += _flush_shard(buf, buildings, split, group_rel,
                                    group_name, shard_idx, True, logger)
            buf, buildings = _empty_buf(), []
            shard_idx += 1
            shards_written += 1
            multi = True
            logger.snapshot(f"pós-shard {shard_idx - 1}")
    if buf["x"]:
        total_w += _flush_shard(buf, buildings, split, group_rel,
                                group_name, shard_idx, multi, logger)
        shards_written += 1
    bar.close()

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"windows": total_w,
                                  "shards": shards_written}))
    dt = time.time() - t_g0
    logger.term(f"[step2-3] ({idx}/{total_groups}) {split}/{group_rel}: "
                f"{total_w} janelas em {shards_written} shard(s) | "
                f"total {_fmt_dur(dt)}")
    logger.group_end(f"{split}/{group_rel}",
                     f"| janelas={total_w} | {_fmt_dur(dt)}")
    return total_w


def discover():
    plan = []
    for split in CFG.splits:
        base = CFG.split_root / split
        if not base.exists():
            continue
        for gdir in sorted([base] + [p for p in base.rglob("*") if p.is_dir()]):
            files = sorted(f for f in gdir.glob("*.parquet") if f.is_file())
            if files:
                plan.append((split, gdir, files))
    return plan


def run(fine_resume: bool = False, group: str | None = None) -> None:
    logger = RunLogger("step2_3")
    plan = discover()
    if group:
        # LINHA CRUCIAL (execução de teste): mantém apenas grupos cujo
        # caminho relativo contém o filtro — ex.: --group "state=AL",
        # --group "HRSA/11", --group SynD. Case-insensitive.
        g = group.lower()
        plan = [(s, d, f) for (s, d, f) in plan
                if g in str(d.relative_to(CFG.split_root / s)).lower()]
        logger.term(f"[step2-3] filtro --group '{group}': "
                    f"{len(plan)} grupo(s) selecionado(s)")
    logger.term(f"[step2-3] inventário: {len(plan)} grupos, "
                f"{sum(len(f) for *_, f in plan)} arquivos, splits={CFG.splits}")
    if not plan:
        logger.close("vazio")
        raise RuntimeError(f"Nada para processar em {CFG.split_root}.")
    t0 = time.time()
    grand = 0
    for i, (split, gdir, files) in enumerate(plan, 1):
        grand += build_group(split, gdir, files, logger, fine_resume,
                             i, len(plan))
    logger.term(f"[step2-3] FIM: {grand} janelas em {_fmt_dur(time.time()-t0)} "
                f"| saída em {CFG.windows_root}")
    logger.close(f"janelas={grand}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine-resume", action="store_true",
                    help="retoma um grupo interrompido pulando os prédios já "
                         "gravados em shards concluídos (via .shards.jsonl)")
    ap.add_argument("--group", type=str, default=None,
                    help="processa só os grupos cujo caminho contém este texto "
                         "(ex.: 'HRSA/11', 'state=AL', 'SynD') — ideal para "
                         "testar num grupo pequeno antes do run completo")
    a = ap.parse_args()
    run(fine_resume=a.fine_resume, group=a.group)