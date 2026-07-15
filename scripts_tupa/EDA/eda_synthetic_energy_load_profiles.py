# -*- coding: utf-8 -*-
"""
EDA — EnergyBench / Dataset_V0.0 / Synthetic-Energy-Load-Profiles (SINTÉTICOS)
==============================================================================

A pasta sintética do EnergyBench é ordens de grandeza maior que a real
(o card do dataset reporta ~31 milhões de edifícios residenciais simulados e
~718,7 bilhões de observações horárias). Portanto, NADA aqui assume que os
dados cabem em memória. Estratégia em três camadas:

  CAMADA A — INVENTÁRIO SEM LEITURA DE DADOS (só metadados Parquet):
     nº de arquivos, tamanho em disco, nº de linhas por arquivo,
     esquema(s) encontrados, partições. Custo ~zero de RAM.

  CAMADA B — ESTATÍSTICAS EM STREAMING (leitura por batches via
     pyarrow.dataset.Scanner): contagem, nulos, min/max, média e variância
     (algoritmo de Welford por chunks, numericamente estável), quantis
     aproximados por reservoir sampling, e perfis agregados
     (hora do dia x dia da semana, mês) acumulados incrementalmente.
     Nunca materializa o dataset completo.

  CAMADA C — AMOSTRA PARA VISUALIZAÇÃO: n_files_sample arquivos e
     n_series_sample séries são amostrados para os mesmos gráficos
     "estilo Ausgrid" do script dos dados reais (perfil diário,
     sazonalidade, distribuição de picos), permitindo COMPARAR
     visualmente sintético vs. real — verificação importante quando o
     sintético é usado para pré-treinar o baseline neural (W-LSTMix).

Estrutura esperada:
  <ROOT>/Synthetic-Energy-Load-Profiles/
      ├── 15min/  ── <subpastas/estados>/**/*.parquet
      └── Hourly/ ── <subpastas/estados>/**/*.parquet

Uso:
  python eda_synthetic_energy_load_profiles.py --root ./Dataset_V0.0 \
      --resolution Hourly --outdir ./eda_outputs_synthetic \
      --max-batches 2000 --n-files-sample 8

Dependências: pyarrow>=12, pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "time", "date", "ts")
ID_CANDIDATES = ("building_id", "buildingid", "building", "bldg_id", "meter_id",
                 "id", "house_id", "household_id", "customer", "customer_id")


def _find(names: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {n.lower().strip().replace(" ", "_"): n for n in names}
    for c in candidates:
        if c in norm:
            return norm[c]
    for k, v in norm.items():
        if any(c in k for c in candidates):
            return v
    return None


# ==========================================================================
# CAMADA A — inventário via metadados (não lê os dados em si)
# ==========================================================================

def inventory(files: list[Path], outdir: Path) -> pd.DataFrame:
    rows, schemas = [], defaultdict(int)
    for f in files:
        try:
            md = pq.ParquetFile(f).metadata
            rows.append({
                "file": str(f), "size_mb": f.stat().st_size / 1e6,
                "n_rows": md.num_rows, "n_cols": md.num_columns,
                "n_row_groups": md.num_row_groups,
            })
            schemas[tuple(md.schema.names)] += 1
        except Exception as e:
            rows.append({"file": str(f), "size_mb": f.stat().st_size / 1e6,
                         "n_rows": None, "n_cols": None,
                         "n_row_groups": None, "error": str(e)})
    inv = pd.DataFrame(rows)
    inv.to_csv(outdir / "inventory_files.csv", index=False)

    total_rows = int(inv["n_rows"].dropna().sum())
    print(f"[A] Inventário: {len(files):,} arquivos | "
          f"{inv['size_mb'].sum()/1024:.1f} GB | {total_rows:,} linhas (metadados)")
    print(f"[A] Esquemas distintos encontrados: {len(schemas)}")
    with open(outdir / "inventory_schemas.json", "w") as fh:
        json.dump({" | ".join(k): v for k, v in schemas.items()}, fh, indent=2)

    # distribuição de tamanho e de linhas por arquivo
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    inv["size_mb"].hist(ax=axes[0], bins=40); axes[0].set_title("Tamanho por arquivo (MB)")
    inv["n_rows"].dropna().hist(ax=axes[1], bins=40); axes[1].set_title("Linhas por arquivo")
    fig.tight_layout(); fig.savefig(outdir / "inventory_hist.png", dpi=130)
    plt.close(fig)
    return inv


# ==========================================================================
# CAMADA B — estatísticas em streaming (Welford + acumuladores de perfil)
# ==========================================================================

class StreamStats:
    """Média/variância incremental (Chan et al. — merge de chunks) por coluna."""
    def __init__(self) -> None:
        self.n = 0; self.mean = 0.0; self.m2 = 0.0
        self.minv = np.inf; self.maxv = -np.inf; self.nulls = 0

    def update(self, arr: np.ndarray, n_nulls: int) -> None:
        self.nulls += n_nulls
        arr = arr[np.isfinite(arr)]
        k = arr.size
        if k == 0:
            return
        m, v = float(arr.mean()), float(arr.var())
        delta = m - self.mean
        tot = self.n + k
        self.mean += delta * k / tot
        self.m2 += v * k + delta**2 * self.n * k / tot
        self.n = tot
        self.minv = min(self.minv, float(arr.min()))
        self.maxv = max(self.maxv, float(arr.max()))

    def result(self) -> dict:
        var = self.m2 / self.n if self.n > 1 else np.nan
        return {"n": self.n, "nulls": self.nulls, "mean": self.mean,
                "std": np.sqrt(var), "min": self.minv, "max": self.maxv}


class Reservoir:
    """Amostra uniforme de tamanho fixo p/ quantis aproximados e histograma."""
    def __init__(self, size: int = 400_000, seed: int = 0) -> None:
        self.size = size; self.buf = np.empty(0); self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, arr: np.ndarray) -> None:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        if self.buf.size < self.size:
            take = min(self.size - self.buf.size, arr.size)
            self.buf = np.concatenate([self.buf, arr[:take]])
            arr = arr[take:]
            self.seen += take
        for x in arr[:: max(1, arr.size // 10_000)]:  # subamostra o excedente
            self.seen += 1
            j = self.rng.integers(0, self.seen)
            if j < self.size:
                self.buf[j] = x


def stream_statistics(files: list[Path], outdir: Path,
                      max_batches: int | None, batch_rows: int) -> None:
    dataset = pads.dataset([str(f) for f in files], format="parquet")
    names = dataset.schema.names
    tcol = _find(names, TIME_CANDIDATES)
    idcol = _find(names, ID_CANDIDATES)
    num_cols = [f.name for f in dataset.schema
                if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)]
    num_cols = [c for c in num_cols if c not in (tcol, idcol)]
    print(f"[B] Coluna temporal: {tcol!r} | id: {idcol!r} | "
          f"{len(num_cols)} colunas numéricas")

    stats = {c: StreamStats() for c in num_cols}
    res = Reservoir()
    # acumuladores de perfil: soma e contagem por (dia-da-semana, hora) e mês
    prof_sum = np.zeros((7, 24)); prof_cnt = np.zeros((7, 24))
    month_sum = np.zeros(12); month_cnt = np.zeros(12)
    ids_seen: set = set()

    cols = ([tcol] if tcol else []) + ([idcol] if idcol else []) + num_cols
    scanner = dataset.scanner(columns=cols, batch_size=batch_rows)
    n_rows = 0
    for i, batch in enumerate(scanner.to_batches()):
        if max_batches and i >= max_batches:
            print(f"[B] Interrompido em max_batches={max_batches} (amostragem).")
            break
        tb = pa.Table.from_batches([batch]).to_pandas()
        n_rows += len(tb)

        if idcol and len(ids_seen) < 2_000_000:
            ids_seen.update(tb[idcol].unique().tolist())

        # carga total do batch = soma das colunas numéricas (p/ perfis)
        vals = tb[num_cols].to_numpy(dtype=float)
        total = np.nansum(vals, axis=1)
        for c in num_cols:
            col = tb[c].to_numpy(dtype=float)
            stats[c].update(col, int(np.isnan(col).sum()))
        res.update(total)

        if tcol is not None:
            ts = pd.to_datetime(tb[tcol], errors="coerce")
            ok = ts.notna() & np.isfinite(total)
            if ok.any():
                dow, hr = ts[ok].dt.dayofweek.values, ts[ok].dt.hour.values
                mo = ts[ok].dt.month.values - 1
                np.add.at(prof_sum, (dow, hr), total[ok.values])
                np.add.at(prof_cnt, (dow, hr), 1)
                np.add.at(month_sum, mo, total[ok.values])
                np.add.at(month_cnt, mo, 1)

        if i % 200 == 0:
            print(f"    batch {i:,} — {n_rows:,} linhas processadas")

    # ---- salvar resultados ----
    per_col = pd.DataFrame({c: s.result() for c, s in stats.items()}).T
    per_col.to_csv(outdir / "streaming_column_stats.csv")
    print(f"[B] {n_rows:,} linhas processadas | "
          f"{len(ids_seen):,} ids únicos vistos (parcial se truncado)")

    q = np.nanquantile(res.buf, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    pd.Series(q, index=["p01", "p05", "p25", "p50", "p75", "p95", "p99"],
              name="carga_total_aprox").to_csv(outdir / "streaming_quantiles.csv")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(res.buf, bins=80)
    ax.set_yscale("log")
    ax.set_title("Distribuição da carga total por observação "
                 f"(reservoir n={res.buf.size:,}, escala log)")
    fig.tight_layout(); fig.savefig(outdir / "streaming_load_hist.png", dpi=130)
    plt.close(fig)

    with np.errstate(invalid="ignore", divide="ignore"):
        prof = prof_sum / prof_cnt
        month = month_sum / month_cnt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    im = axes[0].imshow(prof, aspect="auto", origin="lower")
    axes[0].set_title("Perfil médio: dia da semana x hora")
    axes[0].set_xlabel("hora"); axes[0].set_ylabel("dia (0=seg)")
    fig.colorbar(im, ax=axes[0])
    axes[1].bar(range(1, 13), month)
    axes[1].set_title("Carga média por mês")
    axes[1].set_xlabel("mês")
    fig.tight_layout(); fig.savefig(outdir / "streaming_profiles.png", dpi=130)
    plt.close(fig)


# ==========================================================================
# CAMADA C — amostra visual "estilo Ausgrid" (comparável ao script real)
# ==========================================================================

def sample_visuals(files: list[Path], outdir: Path,
                   n_files: int, n_series: int, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(files, dtype=object),
                        size=min(n_files, len(files)), replace=False)
    frames = []
    for f in chosen:
        df = pd.read_parquet(f)
        tcol = _find(list(df.columns), TIME_CANDIDATES)
        if tcol is None:
            continue
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        idcol = _find([c for c in df.columns if c != tcol], ID_CANDIDATES)
        if idcol is not None:
            ids = df[idcol].dropna().unique()
            ids = rng.choice(ids, size=min(max(1, n_series // len(chosen)),
                                           len(ids)), replace=False)
            sub = df[df[idcol].isin(ids)]
            vcols = sub.select_dtypes("number").columns.difference([idcol])
            sub = sub.assign(_v=sub[vcols].sum(axis=1))
            wide = sub.pivot_table(index=tcol, columns=idcol, values="_v")
            wide.columns = [f"{Path(f).stem}:{c}" for c in wide.columns]
        else:
            wide = df.set_index(tcol).select_dtypes("number")
            keep = rng.choice(wide.columns,
                              size=min(max(1, n_series // len(chosen)),
                                       wide.shape[1]), replace=False)
            wide = wide[keep]
            wide.columns = [f"{Path(f).stem}:{c}" for c in wide.columns]
        frames.append(wide)
    if not frames:
        print("[C] Amostra visual indisponível (esquema não reconhecido).")
        return
    wide = pd.concat(frames, axis=1).sort_index()
    print(f"[C] Amostra: {wide.shape[1]} séries x {wide.shape[0]:,} timestamps")

    # séries individuais em janela curta (slicing à la Ausgrid)
    s = wide.dropna(how="all")
    start = s.index[len(s) // 2].normalize()
    win = s.loc[start: start + pd.Timedelta(days=3)]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    win.iloc[:, : min(5, win.shape[1])].plot(ax=ax, lw=1)
    ax.set_title("Sintético — séries amostradas — janela de 3 dias")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / "sample_series.png", dpi=130)
    plt.close(fig)

    # perfil diário útil x fim de semana da amostra
    m = wide.mean(axis=1).to_frame("v")
    m["hour"] = m.index.hour + m.index.minute / 60.0
    m["weekend"] = m.index.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(9, 4))
    for wk, lab in [(False, "dia útil"), (True, "fim de semana")]:
        g = m[m["weekend"] == wk].groupby("hour")["v"]
        ax.plot(g.mean().index, g.mean().values, label=lab)
        ax.fill_between(g.mean().index, g.quantile(0.25).values,
                        g.quantile(0.75).values, alpha=0.2)
    ax.set_title("Sintético (amostra) — perfil diário médio")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "sample_daily_pattern.png", dpi=130)
    plt.close(fig)

    # distribuição de pico por série amostrada
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    wide.max().hist(ax=ax, bins=40)
    ax.set_title("Sintético (amostra) — pico por série")
    fig.tight_layout(); fig.savefig(outdir / "sample_peak_hist.png", dpi=130)
    plt.close(fig)


# ==========================================================================
# ORQUESTRAÇÃO
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="EDA — EnergyBench (sintéticos, out-of-core)")
    ap.add_argument("--root", type=Path, required=True,
                    help="Pasta Dataset_V0.0 (ou a própria Synthetic-Energy-Load-Profiles)")
    ap.add_argument("--resolution", default="Hourly", choices=["Hourly", "15min"])
    ap.add_argument("--subset", default=None,
                    help="Restringe a uma subpasta (ex.: um estado) para testes")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Limita batches na Camada B (None = varredura completa)")
    ap.add_argument("--batch-rows", type=int, default=262_144)
    ap.add_argument("--n-files-sample", type=int, default=8)
    ap.add_argument("--n-series-sample", type=int, default=24)
    ap.add_argument("--outdir", type=Path, default=Path("eda_outputs_synthetic"))
    args = ap.parse_args()

    base = args.root
    if base.name != "Synthetic-Energy-Load-Profiles":
        base = base / "Synthetic-Energy-Load-Profiles"
    root = base / args.resolution
    if args.subset:
        root = root / args.subset
    if not root.exists():
        raise SystemExit(f"Pasta não encontrada: {root}")

    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"Nenhum .parquet em {root}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"EDA (sintéticos) — {root} — {len(files):,} arquivos")

    inventory(files, args.outdir)                       # Camada A
    stream_statistics(files, args.outdir,               # Camada B
                      args.max_batches, args.batch_rows)
    sample_visuals(files, args.outdir,                  # Camada C
                   args.n_files_sample, args.n_series_sample)
    print(f"\nSaídas em: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
