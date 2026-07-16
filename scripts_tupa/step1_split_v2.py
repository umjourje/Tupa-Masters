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
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from config import CFG

# Candidatos (case-insensitive) para detecção de esquema:
TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "date", "time",
                   "ds", "reading_time", "utc_timestamp", "local_time")
ID_CANDIDATES = ("bldg_id", "building_id", "buildingid", "building", "meter_id",
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


# ---------------------------------------------------------------------------
# Retomada (resume) — manifesto de grupos concluídos
# ---------------------------------------------------------------------------
def _marker_path(group_rel: Path) -> Path:
    safe = str(group_rel).replace("\\", "__").replace("/", "__")
    return CFG.split_root / "_manifest_step1" / f"{safe}.done"


def _clear_partial_outputs(group_rel: Path) -> None:
    """Remove QUALQUER saída parcial do grupo antes de reprocessá-lo —
    garante a semântica 'sobrepõe o que ficou pela metade'."""
    for split in CFG.splits:
        shutil.rmtree(CFG.split_root / split / group_rel, ignore_errors=True)


# ---------------------------------------------------------------------------
# Leitura ECONÔMICA de arquivos wide gigantes (ex.: SDG-1H, SynD-1H):
# lê o SCHEMA via pyarrow e depois UMA coluna por vez — evita carregar o
# arquivo inteiro em RAM (causa provável do OOM/'Killed' após state=WY,
# quando a ordem alfabética chega a SDG/SynD).
# ---------------------------------------------------------------------------
def iter_building_series_file(fp: Path):
    """Como iter_building_series, mas decidindo o layout pelo SCHEMA quando
    possível (parquet), para nunca materializar um wide gigante inteiro."""
    if fp.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            cols = list(pq.ParquetFile(fp).schema_arrow.names)
        except Exception:
            cols = None
        if cols:
            tcol = _find_col(cols, TIME_CANDIDATES)
            non_time = [c for c in cols if c != tcol]
            idcol = _find_col(non_time, ID_CANDIDATES)
            vcol = _find_col(non_time, VALUE_CANDIDATES)
            if idcol is None and vcol is None:
                # WIDE: uma coluna por prédio — lê [tempo, coluna] por vez.
                ts = (pd.to_datetime(
                        pd.read_parquet(fp, columns=[tcol])[tcol],
                        errors="coerce") if tcol else None)
                for c in non_time:
                    col = pd.read_parquet(fp, columns=[c])[c]
                    col = pd.to_numeric(col, errors="coerce")
                    if col.notna().sum() == 0:
                        continue
                    out = pd.DataFrame({"energy": col})
                    if ts is not None:
                        out["timestamp"] = ts.values
                    yield _safe_name(c), out
                return
            if idcol is None and vcol is not None:
                # SINGLE: lê só as colunas necessárias.
                use = [vcol] + ([tcol] if tcol else [])
                df = pd.read_parquet(fp, columns=use)
                yield from iter_building_series(df, fp.stem)
                return
            # LONG: precisa de id+valor(+tempo) — lê apenas essas colunas.
            use = [idcol, vcol] + ([tcol] if tcol else [])
            df = pd.read_parquet(fp, columns=use)
            yield from iter_building_series(df, fp.stem)
            return
    df = _read_any(fp)
    if df is not None and not df.empty:
        yield from iter_building_series(df, fp.stem)


def _leaf_dirs(src: Path) -> list[Path]:
    """Diretórios-folha de dados: todo diretório que contém DIRETAMENTE
    arquivos parquet/csv. Isso generaliza a descoberta para árvores de
    profundidade variável:
      real:      Hourly/Commercial/Prayas/           -> grupo "Commercial/Prayas"
      sintético: Hourly/SynD/                        -> grupo "SynD"
                 Hourly/HRSA/01/                     -> grupo "HRSA/01"
                 Hourly/Buildings-900K/comstock_.../state=AL/
                                    -> grupo "Buildings-900K/comstock_.../state=AL"
    CAVEAT: partições do MESMO prédio devem estar em arquivos do MESMO
    diretório (é o caso do EnergyBench); subpastas viram grupos distintos.
    """
    leaves = []
    for d in sorted([src] + [p for p in src.rglob("*") if p.is_dir()]):
        if any(f.suffix.lower() in (".parquet", ".csv")
               for f in d.iterdir() if f.is_file()):
            leaves.append(d)
    return leaves


def run() -> None:
    src = CFG.raw_root / CFG.resolution
    if not src.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {src}")
    leaves = _leaf_dirs(src)
    print(f"[step1] {len(leaves)} grupos (diretórios-folha) em {src}", flush=True)
    if not leaves:
        raise RuntimeError(f"Nenhum diretório com parquet/csv sob {src}")

    n_ok = n_skip = 0
    for i, leaf in enumerate(leaves, 1):
        group_rel = leaf.relative_to(src)          # ex.: Buildings-900K/.../state=AL
        marker = _marker_path(group_rel)
        # LINHA CRUCIAL (retomada): grupo com marcador .done é PULADO;
        # grupo sem marcador tem qualquer saída parcial APAGADA e é refeito
        # do zero — exatamente "retoma de onde parou, sobrepondo o que
        # ficou pela metade".
        if marker.exists():
            print(f"[step1] ({i}/{len(leaves)}) {group_rel}: "
                  f"pulado (já concluído)", flush=True)
            continue
        _clear_partial_outputs(group_rel)

        # Acumula as PARTES de cada prédio vindas de TODOS os arquivos do
        # diretório-folha (partições por ano/part-files) antes do corte.
        parts = defaultdict(list)
        for fp in sorted(f for f in leaf.iterdir() if f.is_file()):
            if fp.suffix.lower() not in (".parquet", ".csv"):
                continue
            for bname, bdf in iter_building_series_file(fp):
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
                    f"{group_rel}:{bname}"): serie}
            W = CFG.backcast_length + CFG.forecast_length
            for split, part in split_parts.items():
                if len(part) < W:
                    continue
                dst = CFG.split_root / split / group_rel / f"{bname}.parquet"
                dst.parent.mkdir(parents=True, exist_ok=True)
                part.to_parquet(dst, index=False)
            n_ok += 1
        # LINHA CRUCIAL: o marcador só é gravado APÓS todos os parquets do
        # grupo estarem em disco — um kill no meio deixa o grupo sem
        # marcador e ele será refeito integralmente na próxima execução.
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"buildings": len(parts)}))
        print(f"[step1] ({i}/{len(leaves)}) {group_rel}: "
              f"{len(parts)} prédios extraídos", flush=True)
    print(f"[step1] prédios processados: {n_ok} | descartados (curtos): {n_skip}",
          flush=True)
    print(f"[step1] saída em: {CFG.split_root}", flush=True)


if __name__ == "__main__":
    run()
