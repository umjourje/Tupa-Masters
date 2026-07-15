# -*- coding: utf-8 -*-
"""
EDA — EnergyBench / Dataset_V0.0 / Energy-Load-Profiles (EDIFÍCIOS REAIS)
=========================================================================

Análise exploratória no estilo do repositório de referência
"pierre-haessig/ausgrid-solar-data", adaptada à estrutura do EnergyBench
(https://huggingface.co/datasets/ai-iot/EnergyBench).

O estilo Ausgrid consiste em quatro blocos, reproduzidos aqui:
  1. Leitura + reshape para formato "timeseries-friendly":
       - índice   = datetime (amostragem regular)
       - colunas  = MultiIndex (dataset, edifício/canal)
  2. Slicing / visualização:
       - séries de edifícios individuais em janelas curtas (dias)
       - séries de N edifícios aleatórios sobrepostas
  3. Estatísticas do dataset:
       - pico de carga por edifício, energia total/anualizada
       - razão energia/pico (horas equivalentes de plena carga — mesma
         métrica "energy to peak load ratio" usada no repo Ausgrid)
       - distribuições (histogramas) e completude dos dados
  4. Padrões temporais:
       - perfil diário médio (dia útil vs. fim de semana)
       - padrão semanal e sazonal/anual (heatmap mês x hora)
   + 5. Correlação entre edifícios (análogo ao "PV corr" do Ausgrid,
        aqui aplicado à carga).

Estrutura esperada em disco (após baixar o dataset do HuggingFace):
  <ROOT>/Energy-Load-Profiles/
      ├── 15min/ ─┐
      ├── 30min/  ├── <NomeDoDataset>/**/*.parquet
      └── Hourly/─┘

Como os ~60 sub-datasets do EnergyBench NÃO compartilham um esquema único
(alguns são "largos": Timestamp + uma coluna por edifício/circuito; outros
"longos": timestamp, building_id, value), o carregamento é ESQUEMA-ADAPTATIVO.

Uso:
  python eda_real_energy_load_profiles.py --root ./Dataset_V0.0 \
      --resolution Hourly --dataset all --outdir ./eda_outputs_real

Dependências: pandas, numpy, matplotlib, pyarrow  (opcional: seaborn)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # permite rodar sem display (servidor/CI)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# 1. LEITURA E RESHAPE (estilo solarhome.py do repo Ausgrid)
# --------------------------------------------------------------------------

# Nomes candidatos para colunas-chave (minúsculos, comparação case-insensitive)
TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "time", "date", "ts")
ID_CANDIDATES = (
    "building_id", "buildingid", "building", "meter_id", "customer",
    "customer_id", "house_id", "household_id", "id", "lclid", "dataid",
)
VALUE_CANDIDATES = (
    "value", "energy", "consumption", "load", "kwh", "kw", "power",
    "energy_kwh", "power_kw", "mains", "aggregate",
)


def _find_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    """Retorna a primeira coluna cujo nome (normalizado) casa com os candidatos."""
    norm = {c.lower().strip().replace(" ", "_"): c for c in cols}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    # fallback: correspondência parcial (ex.: 'Timestamp_UTC')
    for key, original in norm.items():
        if any(cand in key for cand in candidates):
            return original
    return None


def read_one_dataset(ds_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    """Lê todos os .parquet de um sub-dataset e devolve um DataFrame LARGO:
    índice datetime, uma coluna por edifício/canal ("timeseries-friendly",
    como o reshape do notebook 'Solar home exploration' do Ausgrid).
    """
    files = sorted(ds_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet em {ds_dir}")
    if max_files:
        files = files[:max_files]

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        tcol = _find_col(list(df.columns), TIME_CANDIDATES)
        if tcol is None:
            # timestamp pode estar no índice
            if np.issubdtype(df.index.dtype, np.datetime64):
                df = df.reset_index().rename(columns={df.index.name or "index": "Timestamp"})
                tcol = "Timestamp"
            else:
                raise ValueError(f"Coluna temporal não identificada em {f.name}: {list(df.columns)[:10]}")
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol])

        idcol = _find_col([c for c in df.columns if c != tcol], ID_CANDIDATES)
        if idcol is not None:
            # ---- formato LONGO -> pivot para LARGO ----
            vcol = _find_col([c for c in df.columns if c not in (tcol, idcol)], VALUE_CANDIDATES)
            if vcol is None:
                nums = df.drop(columns=[tcol, idcol]).select_dtypes("number").columns
                if len(nums) == 0:
                    raise ValueError(f"Coluna de valor não identificada em {f.name}")
                vcol = nums[0]
            wide = df.pivot_table(index=tcol, columns=idcol, values=vcol, aggfunc="mean")
        else:
            # ---- formato LARGO: cada coluna numérica é um edifício/canal ----
            wide = df.set_index(tcol).select_dtypes("number")
            # se cada arquivo = um edifício, usa o nome do arquivo como coluna
            if wide.shape[1] == 1:
                wide.columns = [f.stem]
        wide.columns = [str(c) for c in wide.columns]
        frames.append(wide)

    # concat externo: arquivos podem ser partições temporais (empilhar linhas)
    # ou partições por edifício (juntar colunas) — o join externo cobre ambos
    out = pd.concat(frames, axis=0)
    out = out.groupby(level=0).mean()          # resolve duplicatas de timestamp
    out = out.sort_index()
    return out


def infer_step_hours(index: pd.DatetimeIndex) -> float:
    """Passo temporal predominante, em horas (p/ converter potência→energia)."""
    if len(index) < 3:
        return 1.0
    step = index.to_series().diff().median()
    return step.total_seconds() / 3600.0


# --------------------------------------------------------------------------
# 2. SLICING — plots de edifícios individuais e amostras aleatórias
# --------------------------------------------------------------------------

def plot_single_building(wide: pd.DataFrame, name: str, outdir: Path,
                         n_days: int = 3, seed: int = 42) -> None:
    """Análogo ao 'Customer 1 2011-07 01-03.png' do Ausgrid: um edifício,
    janela de poucos dias."""
    rng = np.random.default_rng(seed)
    col = rng.choice(wide.columns)
    s = wide[col].dropna()
    if s.empty:
        return
    start = s.index[len(s) // 2].normalize()
    win = s.loc[start: start + pd.Timedelta(days=n_days)]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    win.plot(ax=ax, lw=1)
    ax.set_title(f"{name} — edifício '{col}' — janela de {n_days} dias")
    ax.set_ylabel("carga")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_single_building.png", dpi=130)
    plt.close(fig)


def plot_random_buildings(wide: pd.DataFrame, name: str, outdir: Path,
                          n: int = 3, n_days: int = 3, seed: int = 1) -> None:
    """Análogo ao 'PV production ... 3 customers' do Ausgrid."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(wide.columns, size=min(n, wide.shape[1]), replace=False)
    sub = wide[cols].dropna(how="all")
    if sub.empty:
        return
    start = sub.index[len(sub) // 2].normalize()
    win = sub.loc[start: start + pd.Timedelta(days=n_days)]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    win.plot(ax=ax, lw=1)
    ax.set_title(f"{name} — {len(cols)} edifícios aleatórios — {n_days} dias")
    ax.set_ylabel("carga")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_random_buildings.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. ESTATÍSTICAS (pico, energia, razão energia/pico, completude)
# --------------------------------------------------------------------------

def dataset_statistics(wide: pd.DataFrame, name: str, outdir: Path) -> pd.DataFrame:
    step_h = infer_step_hours(wide.index)
    span_days = max((wide.index[-1] - wide.index[0]).days, 1)

    peak = wide.max()
    mean = wide.mean()
    energy = wide.sum() * step_h                       # ~kWh se unidade for kW
    energy_yr = energy * 365.0 / span_days             # anualizada
    # "energy to peak load ratio" (horas/ano), métrica central do repo Ausgrid
    hours_eq = energy_yr / peak.replace(0, np.nan)
    completeness = wide.notna().mean()

    stats = pd.DataFrame({
        "peak": peak, "mean": mean,
        "energy_total": energy, "energy_annualized": energy_yr,
        "energy_to_peak_h_per_year": hours_eq,
        "completeness": completeness,
        "n_obs": wide.notna().sum(),
    })
    stats.to_csv(outdir / f"{name}_building_stats.csv")

    # histogramas (como as distribuições de kWp e pico do Ausgrid)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    peak.hist(ax=axes[0], bins=40); axes[0].set_title("Pico de carga por edifício")
    hours_eq.dropna().hist(ax=axes[1], bins=40)
    axes[1].set_title("Razão energia/pico (h/ano)")
    completeness.hist(ax=axes[2], bins=40); axes[2].set_title("Completude (fração não-nula)")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_distributions.png", dpi=130)
    plt.close(fig)

    # matriz de disponibilidade (edifício x tempo) — controle de qualidade
    avail = wide.notna().resample("W").mean().T
    fig, ax = plt.subplots(figsize=(11, max(2.5, min(8, wide.shape[1] * 0.05))))
    ax.imshow(avail.values, aspect="auto", interpolation="nearest",
              extent=[0, avail.shape[1], 0, avail.shape[0]], vmin=0, vmax=1)
    ax.set_title(f"{name} — disponibilidade semanal (claro = completo)")
    ax.set_xlabel("semanas"); ax.set_ylabel("edifícios")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_availability.png", dpi=130)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
# 4. PADRÕES TEMPORAIS (Pattern_daily_consumption / Pattern_yearly do Ausgrid)
# --------------------------------------------------------------------------

def plot_patterns(wide: pd.DataFrame, name: str, outdir: Path) -> None:
    agg = wide.mean(axis=1).rename("mean_load")  # carga média entre edifícios

    # 4a. perfil diário: dia útil vs. fim de semana, com banda interquartil
    df = agg.to_frame()
    df["hour"] = df.index.hour + df.index.minute / 60.0
    df["weekend"] = df.index.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(9, 4))
    for wk, label in [(False, "dia útil"), (True, "fim de semana")]:
        g = df[df["weekend"] == wk].groupby("hour")["mean_load"]
        m, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(m.index, m.values, label=label)
        ax.fill_between(m.index, q1.values, q3.values, alpha=0.2)
    ax.set_title(f"{name} — perfil diário médio (banda = interquartil)")
    ax.set_xlabel("hora do dia"); ax.set_ylabel("carga média")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_daily_pattern.png", dpi=130)
    plt.close(fig)

    # 4b. padrão anual/sazonal: heatmap mês x hora
    piv = df.pivot_table(index=df.index.month, columns="hour",
                         values="mean_load", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower",
                   extent=[0, 24, 0.5, 12.5])
    ax.set_title(f"{name} — sazonalidade (mês x hora)")
    ax.set_xlabel("hora do dia"); ax.set_ylabel("mês")
    fig.colorbar(im, ax=ax, label="carga média")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_yearly_pattern.png", dpi=130)
    plt.close(fig)

    # 4c. energia diária agregada ao longo do tempo (Pattern_yearly)
    step_h = infer_step_hours(wide.index)
    daily = (wide.sum(axis=1) * step_h).resample("D").sum()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    daily.plot(ax=ax, lw=0.8)
    daily.rolling(30, center=True).mean().plot(ax=ax, lw=2, label="média móvel 30d")
    ax.set_title(f"{name} — energia diária agregada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_daily_energy.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. CORRELAÇÃO ENTRE EDIFÍCIOS (análogo ao "PV corr" do Ausgrid)
# --------------------------------------------------------------------------

def plot_correlation(wide: pd.DataFrame, name: str, outdir: Path,
                     max_buildings: int = 60, seed: int = 7) -> None:
    if wide.shape[1] < 2:
        return
    cols = wide.columns
    if len(cols) > max_buildings:
        rng = np.random.default_rng(seed)
        cols = rng.choice(cols, size=max_buildings, replace=False)
    corr = wide[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"{name} — correlação entre edifícios (amostra ≤{max_buildings})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_correlation.png", dpi=130)
    plt.close(fig)

    off = corr.values[np.triu_indices_from(corr.values, k=1)]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(off, bins=40)
    ax.set_title(f"{name} — distribuição das correlações par-a-par "
                 f"(mediana={np.nanmedian(off):.2f})")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_correlation_hist.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# ORQUESTRAÇÃO
# --------------------------------------------------------------------------

def run_dataset(ds_dir: Path, outdir: Path, max_files: int | None) -> dict | None:
    name = ds_dir.name
    try:
        wide = read_one_dataset(ds_dir, max_files=max_files)
    except Exception as e:  # esquema inesperado: registrar e seguir
        print(f"  [AVISO] {name}: falha na leitura ({e})")
        return None
    if wide.empty or wide.shape[1] == 0:
        print(f"  [AVISO] {name}: vazio após reshape")
        return None

    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: {wide.shape[0]:,} timestamps x {wide.shape[1]:,} edifícios "
          f"({wide.index.min()} → {wide.index.max()})")

    plot_single_building(wide, name, d)
    plot_random_buildings(wide, name, d)
    stats = dataset_statistics(wide, name, d)
    plot_patterns(wide, name, d)
    plot_correlation(wide, name, d)

    step_h = infer_step_hours(wide.index)
    return {
        "dataset": name,
        "n_buildings": wide.shape[1],
        "n_timestamps": wide.shape[0],
        "n_obs": int(wide.notna().sum().sum()),
        "start": wide.index.min(), "end": wide.index.max(),
        "step_hours": step_h,
        "median_peak": float(stats["peak"].median()),
        "median_energy_to_peak_h": float(stats["energy_to_peak_h_per_year"].median()),
        "median_completeness": float(stats["completeness"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA — EnergyBench (edifícios reais)")
    ap.add_argument("--root", type=Path, required=True,
                    help="Pasta Dataset_V0.0 (ou a própria Energy-Load-Profiles)")
    ap.add_argument("--resolution", default="Hourly",
                    choices=["Hourly", "30min", "15min"])
    ap.add_argument("--dataset", default="all",
                    help="'all' ou nome de um sub-dataset (ex.: Enernoc)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limita nº de parquets por dataset (teste rápido)")
    ap.add_argument("--outdir", type=Path, default=Path("eda_outputs_real"))
    args = ap.parse_args()

    base = args.root
    if base.name != "Energy-Load-Profiles":
        base = base / "Energy-Load-Profiles"
    res_dir = base / args.resolution
    if not res_dir.exists():
        raise SystemExit(f"Pasta não encontrada: {res_dir}")

    if args.dataset == "all":
        ds_dirs = sorted(p for p in res_dir.iterdir() if p.is_dir())
    else:
        ds_dirs = [res_dir / args.dataset]

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"EDA (dados reais) — resolução {args.resolution} — {len(ds_dirs)} dataset(s)")

    summary = []
    for ds in ds_dirs:
        row = run_dataset(ds, args.outdir, args.max_files)
        if row:
            summary.append(row)

    if summary:
        pd.DataFrame(summary).to_csv(args.outdir / "SUMMARY_real.csv", index=False)
        print(f"\nResumo geral salvo em {args.outdir/'SUMMARY_real.csv'}")


if __name__ == "__main__":
    main()
