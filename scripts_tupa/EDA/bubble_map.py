# -*- coding: utf-8 -*-
"""
Mapa de bolhas — densidade de observações por localização (EnergyBench)
=======================================================================

Deriva do "Metadata-dataset.csv" (pasta Dataset_V0.0/metadata) um CSV com
lat/lon + nº de observações por sub-dataset ou por localização, e gera um
bubble plot mundial cuja área da bolha codifica a quantidade de observações
(ex.: GoiEner/Espanha, com 632 mi de obs, domina o mapa; escala log opcional).

Geocodificação OFFLINE: dicionário embutido cobrindo todas as localizações do
CSV do EnergyBench (países via iso3 + locais subnacionais como Portland,
Cambridge, Sharjah, Califórnia). Sem chamadas a APIs externas — reprodutível.

Saídas:
  - <outdir>/map_data.csv .......... CSV derivado (Alias, Location, lat, lon,
                                     n_obs, n_buildings, Type)
  - <outdir>/bubble_map.html ....... mapa interativo (plotly, se instalado)
  - <outdir>/bubble_map.png ........ mapa estático (matplotlib, sempre)

Uso:
  python bubble_map_observations.py --metadata Metadata-dataset.csv \
      --by location --size-scale log --outdir ./eda_outputs_map

  --by dataset   -> uma bolha por sub-dataset (com leve jitter quando vários
                    compartilham o mesmo país)
  --by location  -> agrega as observações por localização (padrão)

Dependências: pandas, numpy, matplotlib  (opcional: plotly p/ HTML interativo)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Geocodificação offline
# --------------------------------------------------------------------------
# Centróides aproximados de país (iso3) — suficientes p/ bubble map mundial.
ISO_COORDS = {
    "USA": (39.8, -98.6), "IND": (22.4, 79.0), "CHN": (35.0, 103.8),
    "KOR": (36.4, 127.9), "CHE": (46.8, 8.2), "THA": (15.1, 101.0),
    "MYS": (3.8, 109.7), "PRT": (39.6, -8.0), "GBR": (54.0, -2.5),
    "ARE": (24.0, 54.0), "ZAF": (-29.0, 25.1), "POL": (52.1, 19.4),
    "AUS": (-25.7, 134.5), "DEU": (51.1, 10.4), "ESP": (40.2, -3.6),
    "IRL": (53.2, -8.1), "GRC": (39.1, 22.9), "JPN": (36.6, 138.0),
    "CAN": (56.1, -106.3), "ITA": (42.8, 12.1), "NOR": (64.6, 12.7),
    "FRA": (46.6, 2.5), "SVK": (48.7, 19.5), "MEX": (23.9, -102.5),
    "CRI": (9.9, -84.2), "LKA": (7.7, 80.7), "AUT": (47.6, 14.1),
}
# Localizações subnacionais/específicas presentes no CSV do EnergyBench
# (a chave é comparada em minúsculas, por 'contém').
PLACE_COORDS = {
    "portland": (45.52, -122.68),            # NEEA
    "cambridge": (52.20, 0.12),              # ULE (Cambridge, GBR)
    "sharjah": (25.35, 55.42),               # IOT
    "phoenix": (33.45, -112.07),             # HB
    "california": (36.78, -119.42),          # Honda SMART Home
    "british columbia": (53.73, -127.65),    # HUE
    "scotland": (56.49, -4.20),              # NESEMP
    "great britain": (54.00, -2.50),         # UKST
    "southern china": (23.13, 113.26),       # EWELD/IPC (Guangdong aprox.)
    "sceaux": (48.78, 2.29),                 # IHEPC
    "sri lanka": (7.70, 80.70),              # RSL
    "italy, austria": (46.60, 13.85),        # GREEND (fronteira ITA/AUT)
    "usa, europe": (39.8, -98.6),            # PES (multi-região -> EUA)
}


def geocode(location: str, iso: str) -> tuple[float, float] | tuple[None, None]:
    loc = str(location).strip().lower()
    for key, (la, lo) in PLACE_COORDS.items():
        if key in loc:
            return la, lo
    iso = str(iso).strip().upper()
    if iso in ISO_COORDS:
        return ISO_COORDS[iso]
    return None, None


# --------------------------------------------------------------------------
# Derivação do CSV
# --------------------------------------------------------------------------

def build_map_data(metadata_csv: Path, by: str) -> pd.DataFrame:
    md = pd.read_csv(metadata_csv)
    md.columns = [c.strip() for c in md.columns]

    def _num(x):
        try:
            return float(str(x).replace(",", ""))
        except (ValueError, TypeError):
            return np.nan

    df = pd.DataFrame({
        "Alias": md["Alias"],
        "Type": md["Type"],
        "Location": md["Location"],
        "iso": md["iso"],
        "n_obs": md["No. of Obs"].map(_num),
        "n_buildings": md["No. of Buildings"].map(_num),
    })
    coords = df.apply(lambda r: geocode(r["Location"], r["iso"]), axis=1)
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]

    missing = df[df["lat"].isna()]
    if len(missing):
        print("[AVISO] Localizações sem coordenadas (adicione em PLACE/ISO_COORDS):")
        print(missing[["Alias", "Location", "iso"]].to_string(index=False))
    df = df.dropna(subset=["lat", "lon", "n_obs"])

    if by == "location":
        df = (df.groupby(["Location", "iso", "lat", "lon"], as_index=False)
                .agg(n_obs=("n_obs", "sum"),
                     n_buildings=("n_buildings", "sum"),
                     n_datasets=("Alias", "count"),
                     datasets=("Alias", lambda s: "; ".join(s))))
        df["label"] = df["Location"]
        # localizações distintas com o mesmo centróide (ex.: 'UK' e
        # 'Great Britain') recebem leve deslocamento p/ não se sobreporem
        rng = np.random.default_rng(0)
        dup = df.duplicated(subset=["lat", "lon"], keep=False)
        df.loc[dup, "lat"] += rng.uniform(-1.5, 1.5, dup.sum())
        df.loc[dup, "lon"] += rng.uniform(-1.5, 1.5, dup.sum())
    else:  # by == "dataset": jitter leve p/ bolhas no mesmo país
        rng = np.random.default_rng(0)
        dup = df.duplicated(subset=["lat", "lon"], keep=False)
        df.loc[dup, "lat"] += rng.uniform(-1.5, 1.5, dup.sum())
        df.loc[dup, "lon"] += rng.uniform(-1.5, 1.5, dup.sum())
        df["label"] = df["Alias"]
    return df.sort_values("n_obs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _bubble_sizes(n_obs: pd.Series, scale: str, max_pt: float = 2500.0) -> np.ndarray:
    v = n_obs.to_numpy(dtype=float)
    if scale == "log":
        w = np.log10(np.clip(v, 1, None))
    else:
        w = v
    return max_pt * (w / w.max()) ** 2 + 15  # área ∝ peso²; piso p/ visibilidade


def plot_matplotlib(df: pd.DataFrame, outdir: Path, scale: str) -> None:
    sizes = _bubble_sizes(df["n_obs"], scale)
    colors = df["Type"].map({"Commercial": "tab:orange"}).fillna("tab:blue") \
        if "Type" in df.columns else "tab:blue"

    fig, ax = plt.subplots(figsize=(13, 6.5))
    # moldura mundial simples (grade equiretangular); sem shapefiles externos
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-60, 81, 20))
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax.scatter(df["lon"], df["lat"], s=sizes, c=colors,
               alpha=0.55, edgecolors="k", linewidths=0.5, zorder=3)
    # rótulos apenas nas maiores bolhas, p/ não poluir
    for _, r in df.head(12).iterrows():
        ax.annotate(f"{r['label']}\n{r['n_obs']/1e6:.1f}M",
                    (r["lon"], r["lat"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("EnergyBench — densidade de observações por localização "
                 f"(área da bolha ∝ nº de observações, escala {scale})")
    if "Type" in df.columns:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color="tab:blue", label="Residential"),
            Line2D([], [], marker="o", ls="", color="tab:orange", label="Commercial"),
        ], loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "bubble_map.png", dpi=150)
    plt.close(fig)
    print(f"Mapa estático: {outdir/'bubble_map.png'}")


def plot_plotly(df: pd.DataFrame, outdir: Path, scale: str) -> bool:
    try:
        import plotly.express as px
    except ImportError:
        print("[INFO] plotly não instalado — apenas o PNG foi gerado "
              "(pip install plotly para o mapa interativo).")
        return False
    size = np.log10(np.clip(df["n_obs"], 1, None)) if scale == "log" else df["n_obs"]
    fig = px.scatter_geo(
        df.assign(_size=size),
        lat="lat", lon="lon", size="_size", size_max=45,
        color="Type" if "Type" in df.columns else None,
        hover_name="label",
        hover_data={"n_obs": ":,", "n_buildings": ":,",
                    "lat": False, "lon": False, "_size": False},
        projection="natural earth",
        title="EnergyBench — densidade de observações por localização "
              f"(bolha ∝ nº de observações, escala {scale})",
    )
    fig.write_html(outdir / "bubble_map.html", include_plotlyjs="cdn")
    print(f"Mapa interativo: {outdir/'bubble_map.html'}")
    return True


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Bubble map de observações (EnergyBench)")
    ap.add_argument("--metadata", type=Path, required=True,
                    help="Caminho do Metadata-dataset.csv")
    ap.add_argument("--by", choices=["location", "dataset"], default="location")
    ap.add_argument("--size-scale", choices=["log", "linear"], default="log",
                    help="'log' recomendado: as obs variam de ~7e2 a ~6e8")
    ap.add_argument("--outdir", type=Path, default=Path("eda_outputs_map"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = build_map_data(args.metadata, args.by)
    df.to_csv(args.outdir / "map_data.csv", index=False)
    print(f"CSV derivado ({len(df)} linhas): {args.outdir/'map_data.csv'}")
    print(df.head(10)[["label", "lat", "lon", "n_obs"]].to_string(index=False))

    plot_matplotlib(df, args.outdir, args.size_scale)
    plot_plotly(df, args.outdir, args.size_scale)


if __name__ == "__main__":
    main()