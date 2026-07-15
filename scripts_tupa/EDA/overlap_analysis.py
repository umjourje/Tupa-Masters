# -*- coding: utf-8 -*-
"""
Sobreposição temporal entre datasets — EnergyBench (Metadata-dataset.csv)
=========================================================================

Identifica as janelas de tempo em que o maior número de sub-datasets do
EnergyBench coleta dados simultaneamente, a partir das colunas Start/End
do CSV de metadados.

MÉTODO (sweep line):
  1. Cada dataset vira um intervalo [Start, End].
  2. Ordenam-se todos os eventos de início (+1) e fim (-1).
  3. Entre dois eventos consecutivos, o conjunto de datasets ativos é
     constante -> cada trecho é uma "interseção candidata" com seu conjunto
     exato de datasets.
  4. Ranking: nº de datasets (desc) e, em empate, duração (desc).
     Filtros: --min-days (duração mínima) e --distinct (remove conjuntos
     que são subconjuntos de um já ranqueado — evita top-10 dominado por
     variações aninhadas do mesmo pico).

SAÍDAS:
  - <outdir>/overlap_top.csv ......... top-k interseções: rank, período,
        duração, nº de datasets, lista de aliases, envelope (mín início /
        máx fim do grupo)
  - <outdir>/overlap_segments.csv .... todos os segmentos da sweep line
  - <outdir>/overlap_boxplot.png ..... gráfico de intervalos: por interseção,
        linha fina = envelope do grupo (mín início -> máx fim);
        barra grossa = a janela de interseção; rótulo = n datasets
  - <outdir>/coverage_curve.png ...... nº de datasets ativos ao longo do tempo

USO:
  python overlap_analysis.py --metadata Metadata-dataset.csv \
      --top 10 --min-days 90 --distinct --outdir ./eda_outputs_overlap

  Filtros opcionais: --type Residential|Commercial  --resolution-only 1H etc.

Dependências: pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------------------------------
# 1. Leitura dos intervalos
# --------------------------------------------------------------------------

def load_intervals(metadata_csv: Path, type_filter: str | None) -> pd.DataFrame:
    md = pd.read_csv(metadata_csv)
    md.columns = [c.strip() for c in md.columns]
    df = pd.DataFrame({
        "alias": md["Alias"].astype(str),
        "type": md.get("Type"),
        "start": pd.to_datetime(md["Start"], errors="coerce"),
        "end": pd.to_datetime(md["End"], errors="coerce"),
    })
    bad = df[df["start"].isna() | df["end"].isna() | (df["end"] <= df["start"])]
    if len(bad):
        print("[AVISO] Intervalos inválidos ignorados:",
              ", ".join(bad["alias"].tolist()))
    df = df.drop(index=bad.index)
    if type_filter:
        df = df[df["type"].str.lower() == type_filter.lower()]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Sweep line -> segmentos de cobertura constante
# --------------------------------------------------------------------------

def sweep_segments(iv: pd.DataFrame) -> pd.DataFrame:
    """Fatia a linha do tempo em segmentos maximais com conjunto constante
    de datasets ativos. Retorna um DataFrame com um segmento por linha."""
    events = []  # (tempo, +1/-1, alias) — fim antes de início no mesmo t
    for _, r in iv.iterrows():
        events.append((r["start"], 1, r["alias"]))
        events.append((r["end"], -1, r["alias"]))
    events.sort(key=lambda e: (e[0], e[1]))

    segments, active = [], set()
    prev_t = None
    for t, delta, alias in events:
        if prev_t is not None and t > prev_t and active:
            segments.append({
                "start": prev_t, "end": t,
                "n_datasets": len(active),
                "datasets": tuple(sorted(active)),
            })
        active.add(alias) if delta == 1 else active.discard(alias)
        prev_t = t

    seg = pd.DataFrame(segments)
    if seg.empty:
        return seg
    # funde segmentos consecutivos com o MESMO conjunto (eventos coincidentes)
    merged = [seg.iloc[0].to_dict()]
    for _, r in seg.iloc[1:].iterrows():
        if r["datasets"] == merged[-1]["datasets"] and r["start"] == merged[-1]["end"]:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r.to_dict())
    seg = pd.DataFrame(merged)
    seg["duration_days"] = (seg["end"] - seg["start"]).dt.days
    return seg


# --------------------------------------------------------------------------
# 3. Ranking do top-k
# --------------------------------------------------------------------------

def rank_overlaps(seg: pd.DataFrame, iv: pd.DataFrame, top: int,
                  min_days: int, distinct: bool) -> pd.DataFrame:
    cand = seg[seg["duration_days"] >= min_days].copy()
    cand = cand.sort_values(["n_datasets", "duration_days"],
                            ascending=[False, False])
    lookup = iv.set_index("alias")[["start", "end"]]

    rows, kept_sets = [], []
    for _, r in cand.iterrows():
        s = set(r["datasets"])
        if distinct and any(s <= k for k in kept_sets):   # subconjunto de já-ranqueado
            continue
        kept_sets.append(s)
        members = lookup.loc[list(s)]
        rows.append({
            "rank": len(rows) + 1,
            "overlap_start": r["start"], "overlap_end": r["end"],
            "overlap_days": r["duration_days"],
            "overlap_years": round(r["duration_days"] / 365.25, 2),
            "n_datasets": r["n_datasets"],
            "group_min_start": members["start"].min(),
            "group_max_end": members["end"].max(),
            "datasets": "; ".join(sorted(s)),
        })
        if len(rows) >= top:
            break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Gráficos
# --------------------------------------------------------------------------

def plot_overlap_boxplot(topdf: pd.DataFrame, outdir: Path) -> None:
    """Estilo 'boxplot' de intervalos: por interseção (eixo y),
    linha fina = envelope do grupo (mín início -> máx fim dos membros);
    barra grossa = a janela de interseção em si."""
    n = len(topdf)
    fig, ax = plt.subplots(figsize=(11, 0.6 * n + 2))
    for i, r in topdf.iterrows():
        y = n - 1 - i
        # "whiskers": envelope do grupo
        ax.plot([r["group_min_start"], r["group_max_end"]], [y, y],
                color="gray", lw=1.5, solid_capstyle="butt", zorder=2)
        for x in (r["group_min_start"], r["group_max_end"]):   # caps
            ax.plot([x, x], [y - 0.18, y + 0.18], color="gray", lw=1.5)
        # "caixa": interseção
        ax.barh(y, (r["overlap_end"] - r["overlap_start"]),
                left=r["overlap_start"], height=0.5,
                color="tab:blue", alpha=0.85, zorder=3)
        ax.annotate(f"n={r['n_datasets']} | {r['overlap_years']:.1f} a",
                    (r["overlap_end"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"top{r['rank']}" for _, r in
                        topdf.iloc[::-1].iterrows()])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Tempo")
    ax.set_title("Interseções temporais entre datasets — barra = janela de "
                 "interseção; linha = envelope do grupo (mín início → máx fim)")
    fig.tight_layout()
    fig.savefig(outdir / "overlap_boxplot.png", dpi=150)
    plt.close(fig)


def plot_coverage(seg: pd.DataFrame, outdir: Path) -> None:
    """Curva do nº de datasets ativos ao longo do tempo (função-escada)."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs, ys = [], []
    for _, r in seg.iterrows():
        xs += [r["start"], r["end"]]
        ys += [r["n_datasets"], r["n_datasets"]]
    ax.plot(xs, ys, lw=1)
    ax.fill_between(xs, ys, alpha=0.2, step=None)
    peak = seg.loc[seg["n_datasets"].idxmax()]
    ax.annotate(f"pico: {peak['n_datasets']} datasets",
                (peak["start"], peak["n_datasets"]),
                xytext=(8, -12), textcoords="offset points", fontsize=8)
    ax.set_ylabel("datasets ativos")
    ax.set_xlabel("Tempo")
    ax.set_title("Cobertura temporal — nº de datasets coletando simultaneamente")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "coverage_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Sobreposição temporal (EnergyBench)")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-days", type=int, default=30,
                    help="Duração mínima da interseção (dias) p/ ranquear")
    ap.add_argument("--distinct", action="store_true",
                    help="Suprime interseções cujo conjunto é subconjunto "
                         "de uma já ranqueada (evita picos aninhados)")
    ap.add_argument("--type", default=None, choices=["Residential", "Commercial"],
                    help="Filtra por categoria antes da análise")
    ap.add_argument("--outdir", type=Path, default=Path("eda_outputs_overlap"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    iv = load_intervals(args.metadata, args.type)
    print(f"{len(iv)} datasets com intervalos válidos "
          f"({iv['start'].min():%Y-%m-%d} → {iv['end'].max():%Y-%m-%d})")

    seg = sweep_segments(iv)
    seg.assign(datasets=seg["datasets"].map(lambda t: "; ".join(t))) \
       .to_csv(args.outdir / "overlap_segments.csv", index=False)

    topdf = rank_overlaps(seg, iv, args.top, args.min_days, args.distinct)
    topdf.to_csv(args.outdir / "overlap_top.csv", index=False)

    print(f"\nTop {len(topdf)} interseções "
          f"(min {args.min_days} dias{', distintas' if args.distinct else ''}):")
    for _, r in topdf.iterrows():
        print(f"  top{r['rank']}: {r['n_datasets']} datasets | "
              f"{r['overlap_start']:%Y-%m-%d} → {r['overlap_end']:%Y-%m-%d} "
              f"({r['overlap_years']:.1f} anos)")

    plot_overlap_boxplot(topdf, args.outdir)
    plot_coverage(seg, args.outdir)
    print(f"\nSaídas em: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()