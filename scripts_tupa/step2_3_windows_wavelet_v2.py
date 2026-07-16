"""
PASSOS 2 e 3 — Janelamento (overlapping sliding window) por grupo/país e
decomposição wavelet POR JANELA, salva em formato nativo do PyTorch (.pt).

Diferença fundamental em relação ao train.py original do W-LSTMix:
  * Original: `decompose_series(series)` roda sobre a SÉRIE INTEIRA no
    __init__ do Dataset; cada janela recebe um trend que "enxergou" o futuro.
  * Aqui: `pywt.wavedec` roda sobre CADA JANELA isoladamente. A janela i
    nunca usa amostras fora de [start, start+backcast+forecast). É isso que
    elimina o data leak que você identificou no TRY1.

As mesmas funções (`decompose_window`, `iter_windows`) são importadas pelo
passo 8 para reproduzir o processo em tempo de execução no teste.

Visibilidade de execução:
  * Barra tqdm por grupo (prédios) com contagem de janelas em tempo real;
  * Diagnóstico ANTES de processar: quantos splits/setores/grupos/arquivos
    foram encontrados — se estiver tudo zerado, o problema é de caminho
    (CFG.split_root), e o script avisa em vez de terminar em silêncio;
  * Resumo final com janelas totais, tempo e destino dos .pt.
"""
from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
import pywt
import torch
from pathlib import Path
from config import CFG

try:
    from tqdm import tqdm
except ImportError:                              # fallback sem dependência
    def tqdm(it, **kw):
        return it


def log(msg: str) -> None:
    """print imediato (flush) — evita a sensação de 'nada acontecendo'
    quando stdout está bufferizado (nohup, slurm, redirecionamento)."""
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Passo 3 (função-núcleo): wavelet restrita à janela
# --------------------------------------------------------------------------
def decompose_window(window: np.ndarray,
                     wavelet: str = CFG.wavelet,
                     level: int = CFG.wavelet_level) -> tuple[np.ndarray, np.ndarray]:
    """Decompõe UMA janela em (trend, seasonal+residual) via DWT.

    Mesma lógica do decompose_series(method='wavelet') do repo original,
    porém aplicada só ao trecho janelado.
    """
    # LINHA CRUCIAL: o nível máximo depende do comprimento da JANELA
    # (168 pontos com db4 => máx. 5). Truncamos para evitar warning/erro.
    max_lv = pywt.dwt_max_level(len(window), pywt.Wavelet(wavelet).dec_len)
    lv = min(level, max_lv)

    coeffs = pywt.wavedec(window, wavelet, level=lv)
    # LINHA CRUCIAL: mantém apenas a aproximação (cA_n) e zera todos os
    # detalhes -> reconstrução é o TREND "limpo" da janela.
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet)[: len(window)]
    season = window - trend
    return trend.astype(np.float32), np.nan_to_num(season).astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-8):
    m, s = float(np.mean(x)), float(np.std(x))
    return ((x - m) / (s + eps)).astype(np.float32), m, s


# --------------------------------------------------------------------------
# Passo 2 (função-núcleo): sliding window com sobreposição
# --------------------------------------------------------------------------
def iter_windows(series: np.ndarray):
    """Gera (start, janela_completa) — janela = backcast + forecast."""
    W = CFG.backcast_length + CFG.forecast_length
    # LINHA CRUCIAL: stride < W  =>  janelas SOBREPOSTAS (overlapping),
    # idêntico ao mecanismo do __getitem__ original (start = idx * stride).
    for start in range(0, len(series) - W + 1, CFG.stride):
        yield start, series[start: start + W]


def n_windows(series_len: int) -> int:
    """Quantas janelas iter_windows vai gerar (para a barra de progresso)."""
    W = CFG.backcast_length + CFG.forecast_length
    return 0 if series_len < W else (series_len - W) // CFG.stride + 1


# --------------------------------------------------------------------------
# Diagnóstico prévio: o que existe para processar?
# --------------------------------------------------------------------------
def discover() -> list[tuple[str, Path, list[Path]]]:
    """Lista (split, group_dir, arquivos) por descoberta RECURSIVA de
    diretórios-folha (contêm parquet diretamente) — funciona tanto para a
    árvore real (Sector/Grupo) quanto para a sintética (profundidade variável,
    ex.: Buildings-900K/comstock_.../state=AL). Falha ALTO se nada existir."""
    plan = []
    for split in CFG.splits:
        base = CFG.split_root / split
        if not base.exists():
            log(f"[step2-3][AVISO] não existe: {base}")
            continue
        for gdir in sorted([base] + [p for p in base.rglob("*") if p.is_dir()]):
            files = sorted(f for f in gdir.glob("*.parquet") if f.is_file())
            if files:
                plan.append((split, gdir, files))
    total_files = sum(len(f) for *_, f in plan)
    log(f"[step2-3] inventário: {len(plan)} grupos, "
        f"{total_files} arquivos de prédio, splits={CFG.splits}")
    if not plan:
        raise RuntimeError(
            f"Nada para processar em {CFG.split_root}. "
            "Confira CFG.out_root/CFG.resolution e se o passo 1 foi executado.")
    return plan


# --------------------------------------------------------------------------
# Construção dos arquivos .pt por grupo (país/dataset) e por split
# --------------------------------------------------------------------------
def _empty_buf():
    return {k: [] for k in ("x", "trend", "season", "trend_norm",
                            "season_norm", "building_idx", "start", "stats")}


def _flush_shard(buf, buildings, split: str, group_rel: Path,
                 group_name: str, shard_idx: int, multi: bool) -> int:
    """Grava um shard .pt (dict de tensores) e devolve o nº de janelas."""
    if not buf["x"]:
        return 0
    out = {
        "x":           torch.tensor(np.stack(buf["x"])),
        "trend":       torch.tensor(np.stack(buf["trend"])),
        "season":      torch.tensor(np.stack(buf["season"])),
        "trend_norm":  torch.tensor(np.stack(buf["trend_norm"])),
        "season_norm": torch.tensor(np.stack(buf["season_norm"])),
        "building_idx": torch.tensor(buf["building_idx"], dtype=torch.long),
        "start":        torch.tensor(buf["start"], dtype=torch.long),
        "stats":        torch.tensor(buf["stats"]),
        "buildings":    buildings,
        "meta": {"backcast": CFG.backcast_length, "forecast": CFG.forecast_length,
                 "stride": CFG.stride, "wavelet": CFG.wavelet,
                 "level": CFG.wavelet_level, "shard": shard_idx},
    }
    name = f"{group_name}.part{shard_idx:03d}.pt" if multi else f"{group_name}.pt"
    dst = CFG.windows_root / split / group_rel.parent / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)                     # dict janelado+decomposto p/ passo 6
    mb = dst.stat().st_size / 1e6
    log(f"[step2-3]   shard gravado: {dst} "
        f"({out['x'].shape[0]} janelas, {len(buildings)} prédios, {mb:.1f} MB)")
    return int(out["x"].shape[0])


def build_group(split: str, gdir: Path, files: list[Path]) -> int:
    group_rel = gdir.relative_to(CFG.split_root / split)
    group_name = group_rel.name if str(group_rel) != "." else "root"
    # 1º passe barato: estima nº de shards p/ decidir o esquema de nomes.
    multi = False
    est = 0
    for fp in files:
        try:
            est += n_windows(len(pd.read_parquet(fp, columns=["energy"])))
        except Exception:
            pass
        if est > CFG.max_windows_per_shard:
            multi = True
            break

    buf, buildings = _empty_buf(), []
    total_w, shard_idx, shards_written = 0, 0, 0
    bar = tqdm(files, desc=f"{split}/{group_rel}",
               unit="prédio", file=sys.stdout, dynamic_ncols=True)
    for fp in bar:
        s = np.nan_to_num(pd.read_parquet(fp)["energy"]
                          .to_numpy(dtype=np.float64))
        b_local = len(buildings)             # índice LOCAL ao shard corrente
        buildings.append(fp.stem)
        for start, w in iter_windows(s):
            trend, season = decompose_window(w)      # PASSO 3, por janela
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
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(janelas=total_w + len(buf["x"]),
                            shard=shard_idx, ultimo=fp.stem[:16])
        # LINHA CRUCIAL (RAM): descarrega o shard SOMENTE em fronteira de
        # prédio — cada prédio fica inteiro num único shard, garantindo que
        # a fusão de rótulos por prédio (passo 5) nunca cruze shards.
        if len(buf["x"]) >= CFG.max_windows_per_shard:
            total_w += _flush_shard(buf, buildings, split, group_rel,
                                    group_name, shard_idx, multi=True)
            buf, buildings = _empty_buf(), []
            shard_idx += 1
            shards_written += 1
            multi = True
    if buf["x"]:
        total_w += _flush_shard(buf, buildings, split, group_rel,
                                group_name, shard_idx, multi=multi)
        shards_written += 1
    if total_w == 0:
        log(f"[step2-3][AVISO] {split}/{group_rel}: 0 janelas "
            f"(séries menores que {CFG.backcast_length + CFG.forecast_length}?)")
    else:
        log(f"[step2-3] OK {split}/{group_rel}: {total_w} janelas "
            f"em {shards_written} shard(s)")
    return total_w


def run() -> None:
    t0 = time.time()
    plan = discover()
    grand_total = 0
    for i, (split, gdir, files) in enumerate(plan, 1):
        rel = gdir.relative_to(CFG.split_root / split)
        log(f"[step2-3] ({i}/{len(plan)}) processando {split}/{rel} "
            f"({len(files)} prédios)...")
        grand_total += build_group(split, gdir, files)
    log(f"[step2-3] FIM: {grand_total} janelas em {time.time() - t0:.1f}s "
        f"| saída em {CFG.windows_root}")


if __name__ == "__main__":
    run()
