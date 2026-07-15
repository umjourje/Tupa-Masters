"""
PASSO 1 — Extração por prédio + split treino/teste do EnergyBench BRUTO.

Por que este passo foi reescrito:
  O EnergyBench NÃO segue "1 arquivo = 1 prédio". O README oficial diz que
  datasets com muitos prédios foram particionados em múltiplos parquets, e
  na prática coexistem 3 layouts:
    (a) WIDE : coluna de tempo + N colunas numéricas (1 por prédio/circuito)
               ex.: Berkely (mels_S, lig_S, hvac_N, ..., Timestamp)
    (b) LONG : coluna de id do prédio + coluna de tempo + coluna de valor
    (c) SINGLE: uma única série com coluna 'energy' (contrato do W-LSTMix)
  Além disso, o MESMO prédio pode estar repartido em vários arquivos
  (partições por ano) — as partes são concatenadas por id e ordenadas no
  tempo ANTES do corte.

O que este passo entrega:
  out_root/01_splits/<res>/<split>/<Sector>/<Grupo>/<edificio>.parquet
  sempre com UMA coluna 'energy' (e 'timestamp' quando existir) —
  normalizando o contrato para que os passos 2-8 fiquem inalterados.

Anti-leak: nenhuma estatística é calculada aqui; o corte temporal
train/test por prédio é a primeira e única operação.
"""
from __future__ import annotations
import hashlib
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from config import CFG

# Candidatos (case-insensitive) para detecção de esquema:
TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "date", "time",
                   "ds", "reading_time", "utc_timestamp", "local_time")
ID_CANDIDATES = ("building_id", "buildingid", "building", "meter_id",
                 "meterid", "meter", "house_id", "household_id", "house",
                 "home_id", "lclid", "dataid", "customer_id", "user_id",
                 "consumer_id", "id", "site_id", "unit")
VALUE_CANDIDATES = ("energy", "kwh", "kw", "consumption", "load", "value",
                    "power", "usage", "reading", "energy_kwh", "energy_kw")


def _find_col(cols, candidates):
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _find_time_col(df: pd.DataFrame):
    c = _find_col(df.columns, TIME_CANDIDATES)
    if c:
        return c
    for col in df.columns:                      # fallback: dtype datetime
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    return None


def _safe_name(name: str) -> str:
    """Ids de prédio podem conter caracteres inválidos para nome de arquivo."""
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_") or "unnamed"


# ---------------------------------------------------------------------------
# LINHA(S) CRUCIAIS: extrator agnóstico de esquema — devolve
# (nome_do_prédio, DataFrame['timestamp'?, 'energy']) para QUALQUER layout.
# ---------------------------------------------------------------------------
def iter_building_series(df: pd.DataFrame, file_stem: str):
    tcol = _find_time_col(df)
    non_time = [c for c in df.columns if c != tcol]
    idcol = _find_col(non_time, ID_CANDIDATES)
    vcol = _find_col(non_time, VALUE_CANDIDATES)

    if idcol is not None and vcol is not None:
        # (b) LONG: um grupo por prédio.
        for bid, g in df.groupby(idcol, sort=False):
            out = pd.DataFrame({"energy": pd.to_numeric(g[vcol],
                                                        errors="coerce")})
            if tcol:
                out["timestamp"] = pd.to_datetime(g[tcol], errors="coerce")
            yield _safe_name(bid), out.reset_index(drop=True)

    elif vcol is not None and idcol is None:
        # (c) SINGLE: contrato original do W-LSTMix.
        out = pd.DataFrame({"energy": pd.to_numeric(df[vcol],
                                                    errors="coerce")})
        if tcol:
            out["timestamp"] = pd.to_datetime(df[tcol], errors="coerce")
        yield _safe_name(file_stem), out

    else:
        # (a) WIDE: cada coluna numérica é a série de um prédio/circuito.
        num_cols = [c for c in non_time
                    if pd.api.types.is_numeric_dtype(df[c])]
        ts = pd.to_datetime(df[tcol], errors="coerce") if tcol else None
        for c in num_cols:
            out = pd.DataFrame({"energy": pd.to_numeric(df[c],
                                                        errors="coerce")})
            if ts is not None:
                out["timestamp"] = ts.values
            yield _safe_name(c), out


def _read_any(fp: Path) -> pd.DataFrame | None:
    try:
        if fp.suffix.lower() == ".parquet":
            return pd.read_parquet(fp)
        if fp.suffix.lower() == ".csv":
            return pd.read_csv(fp)
    except Exception as e:                       # arquivo corrompido etc.
        print(f"[step1][AVISO] falha ao ler {fp}: {e}")
    return None


def _clean(series_df: pd.DataFrame) -> pd.DataFrame:
    """Ordena no tempo, remove timestamps duplicados e apara NaNs das bordas."""
    df = series_df
    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp")
                .drop_duplicates(subset="timestamp", keep="first")
                .reset_index(drop=True))
    valid = df["energy"].notna()
    if not valid.any():
        return df.iloc[0:0]
    first, last = valid.idxmax(), valid[::-1].idxmax()
    return df.loc[first:last].reset_index(drop=True)


def _temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    i_tr = int(n * CFG.train_frac)
    # Corte temporal PURO train/test (val é rolling-origin no passo 6):
    return {"train": df.iloc[:i_tr].reset_index(drop=True),
            "test":  df.iloc[i_tr:].reset_index(drop=True)}


def _building_split_bucket(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < CFG.train_frac * 100 else "test"


def run() -> None:
    src = CFG.raw_root / CFG.resolution
    n_ok = n_skip = 0
    for sector in CFG.sectors:
        sector_dir = src / sector
        if not sector_dir.exists():
            print(f"[step1][AVISO] não existe: {sector_dir}")
            continue
        for group_dir in sorted(p for p in sector_dir.iterdir() if p.is_dir()):
            # ---------------------------------------------------------------
            # LINHA CRUCIAL: acumula as PARTES de cada prédio vindas de TODOS
            # os arquivos do grupo (partições por ano etc.) antes de qualquer
            # corte — sem isso, um split por arquivo quebraria a cronologia.
            # ---------------------------------------------------------------
            parts = defaultdict(list)
            for fp in sorted(group_dir.rglob("*")):
                if fp.suffix.lower() not in (".parquet", ".csv"):
                    continue
                df = _read_any(fp)
                if df is None or df.empty:
                    continue
                for bname, bdf in iter_building_series(df, fp.stem):
                    parts[bname].append(bdf)

            for bname, chunks in parts.items():
                serie = _clean(pd.concat(chunks, ignore_index=True))
                if len(serie) < CFG.min_series_len:
                    n_skip += 1
                    continue
                if CFG.split_mode == "temporal":
                    split_parts = _temporal_split(serie)
                else:
                    split_parts = {_building_split_bucket(
                        f"{group_dir.name}:{bname}"): serie}
                W = CFG.backcast_length + CFG.forecast_length
                for split, part in split_parts.items():
                    if len(part) < W:
                        continue
                    dst = (CFG.split_root / split / sector / group_dir.name
                           / f"{bname}.parquet")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    part.to_parquet(dst, index=False)
                n_ok += 1
            if parts:
                print(f"[step1] {sector}/{group_dir.name}: "
                      f"{len(parts)} prédios extraídos")
    print(f"[step1] prédios processados: {n_ok} | descartados (curtos): {n_skip}")
    print(f"[step1] saída em: {CFG.split_root}")


if __name__ == "__main__":
    run()
