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
def discover() -> list[tuple[str, str, Path, list[Path]]]:
    """Lista (split, sector, group_dir, arquivos) e loga o inventário.
    Se nada for encontrado, falha ALTO com o caminho esperado."""
    plan = []
    for split in CFG.splits:
        for sector in CFG.sectors:
            base = CFG.split_root / split / sector
            if not base.exists():
                log(f"[step2-3][AVISO] não existe: {base}")
                continue
            for gdir in sorted(p for p in base.iterdir() if p.is_dir()):
                files = sorted(gdir.glob("*.parquet"))
                if files:
                    plan.append((split, sector, gdir, files))
                else:
                    log(f"[step2-3][AVISO] grupo vazio: {gdir}")
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
def build_group(split: str, sector: str, gdir: Path, files: list[Path]) -> int:
    group = gdir.name
    buf = {k: [] for k in
           ("x", "trend", "season", "trend_norm", "season_norm",
            "building_idx", "start", "stats")}
    buildings = []

    # Barra por PRÉDIO, com contagem de janelas acumulada no postfix —
    # é aqui que você "vê" o processamento acontecendo.
    bar = tqdm(files, desc=f"{split}/{sector}/{group}",
               unit="prédio", file=sys.stdout, dynamic_ncols=True)
    total_w = 0
    for b_idx, fp in enumerate(bar):
        s = pd.read_parquet(fp)["energy"].to_numpy(dtype=np.float64)
        s = np.nan_to_num(s)
        buildings.append(fp.stem)
        nw = n_windows(len(s))
        for start, w in iter_windows(s):
            trend, season = decompose_window(w)          # PASSO 3, por janela
            t_n, tm, ts = standardize(trend)
            s_n, sm, ss = standardize(season)
            buf["x"].append(w.astype(np.float32))
            buf["trend"].append(trend)
            buf["season"].append(season)
            buf["trend_norm"].append(t_n)
            buf["season_norm"].append(s_n)
            buf["building_idx"].append(b_idx)
            buf["start"].append(start)
            buf["stats"].append([tm, ts, sm, ss])
        total_w += nw
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(janelas=total_w, ultimo=f"{fp.stem[:18]}({nw})")

    if not buf["x"]:
        log(f"[step2-3][AVISO] {split}/{sector}/{group}: 0 janelas "
            f"(séries menores que {CFG.backcast_length + CFG.forecast_length}?)")
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
                 "stride": CFG.stride, "wavelet": CFG.wavelet, "level": CFG.wavelet_level},
    }
    dst = CFG.windows_root / split / sector / f"{group}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # LINHA CRUCIAL: torch.save de um dict de tensores já janelados e
    # decompostos — o Dataset do passo 6 apenas indexa, sem reprocessar.
    torch.save(out, dst)
    mb = dst.stat().st_size / 1e6
    log(f"[step2-3] OK {split}/{sector}/{group}: "
        f"{out['x'].shape[0]} janelas de {len(buildings)} prédios "
        f"-> {dst} ({mb:.1f} MB)")
    return int(out["x"].shape[0])


def run() -> None:
    t0 = time.time()
    plan = discover()
    grand_total = 0
    for i, (split, sector, gdir, files) in enumerate(plan, 1):
        log(f"[step2-3] ({i}/{len(plan)}) processando "
            f"{split}/{sector}/{gdir.name} ({len(files)} prédios)...")
        grand_total += build_group(split, sector, gdir, files)
    log(f"[step2-3] FIM: {grand_total} janelas em {time.time() - t0:.1f}s "
        f"| saída em {CFG.windows_root}")


if __name__ == "__main__":
    run()