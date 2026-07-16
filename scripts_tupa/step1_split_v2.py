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
import time
import time
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from config import CFG

try:
    from tqdm import tqdm
except ImportError:                              # fallback sem dependência
    def tqdm(it, **kw):
        return it

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
def _fmt_dur(seconds: float) -> str:
    """Formata duração como 1h02m03s / 4m05s / 6.7s."""
    if seconds >= 3600:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m:02d}m{s:02d}s"
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def _read_col(fp: Path, col: str) -> pd.Series:
    """Lê UMA coluna do parquet, tolerante ao caso em que ela foi gravada
    como ÍNDICE do DataFrame original: o schema do pyarrow a lista como
    coluna, mas pd.read_parquet(columns=[col]) a devolve no index —
    df[col] então estoura com KeyError (o bug do 'Timestamp')."""
    df = pd.read_parquet(fp, columns=[col])
    if col in df.columns:
        return df[col].reset_index(drop=True)
    if df.index.name == col:                 # veio como índice simples
        return df.index.to_series().reset_index(drop=True)
    df = df.reset_index()                    # veio em MultiIndex/outros casos
    if col in df.columns:
        return df[col].reset_index(drop=True)
    raise KeyError(f"coluna {col!r} não encontrada em {fp}")


def _read_cols(fp: Path, cols: list[str]) -> pd.DataFrame:
    """Lê múltiplas colunas com a mesma tolerância a coluna-índice."""
    df = pd.read_parquet(fp, columns=cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.reset_index()
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


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
            # Colunas-fantasma de índice do pandas não são prédios:
            cols = [c for c in cols if not c.startswith("__index_level_")]
            tcol = _find_col(cols, TIME_CANDIDATES)
            non_time = [c for c in cols if c != tcol]
            idcol = _find_col(non_time, ID_CANDIDATES)
            vcol = _find_col(non_time, VALUE_CANDIDATES)
            if idcol is None and vcol is None:
                # WIDE: uma coluna por prédio — lê [tempo, coluna] por vez.
                ts = (pd.to_datetime(_read_col(fp, tcol), errors="coerce")
                      if tcol else None)
                # Barra por COLUNA: em arquivos wide gigantes (SynD/SDG),
                # cada coluna é um prédio lido separadamente — sem isso,
                # o arquivo inteiro parece "travado".
                for c in tqdm(non_time, desc=f"  colunas de {fp.name}",
                              unit="prédio", file=sys.stdout,
                              dynamic_ncols=True, leave=False):
                    col = pd.to_numeric(_read_col(fp, c), errors="coerce")
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
                yield from iter_building_series(_read_cols(fp, use), fp.stem)
                return
            # LONG: precisa de id+valor(+tempo) — lê apenas essas colunas.
            use = [idcol, vcol] + ([tcol] if tcol else [])
            yield from iter_building_series(_read_cols(fp, use), fp.stem)
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


def adopt_existing_outputs(leaves: list[Path], src: Path) -> None:
    """Reconcilia saídas de execuções ANTERIORES ao manifesto: grava o
    marcador .done para todo grupo que já tem parquets em 01_splits,
    EXCETO o último deles na ordem de processamento — como a execução é
    sequencial e ordenada, apenas o último grupo com saída pode ter sido
    interrompido no meio; ele é refeito (protocolo de reposição do último).
    """
    with_output = []
    for leaf in leaves:
        rel = leaf.relative_to(src)
        if _marker_path(rel).exists():
            continue                              # já reconhecido
        has_train = any((CFG.split_root / "train" / rel).glob("*.parquet"))
        if has_train:
            with_output.append(rel)
    if not with_output:
        print("[step1][adopt] nenhum grupo sem marcador com saída existente.",
              flush=True)
        return
    *completos, ultimo = with_output
    for rel in completos:
        m = _marker_path(rel)
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({"adopted": True}))
    print(f"[step1][adopt] {len(completos)} grupos adotados como concluídos; "
          f"será REFEITO apenas o último com saída: {ultimo}", flush=True)


def run(adopt: bool = False, fine_resume: bool = False) -> None:
    src = CFG.raw_root / CFG.resolution
    if not src.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {src}")
    leaves = _leaf_dirs(src)
    print(f"[step1] {len(leaves)} grupos (diretórios-folha) em {src}", flush=True)
    if not leaves:
        raise RuntimeError(f"Nenhum diretório com parquet/csv sob {src}")
    if adopt:
        # LINHA CRUCIAL: reconhece saídas de execuções pré-manifesto sem
        # reprocessá-las (checagem barata: existência de parquet no split
        # de treino), refazendo só o último grupo da ordem.
        adopt_existing_outputs(leaves, src)

    t0_script = time.time()
    n_ok = n_skip = 0
    tot_rows_read = tot_rows_written = 0
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
        if not fine_resume:
            _clear_partial_outputs(group_rel)   # padrão: refaz o grupo inteiro
        t_g0 = time.time()

        # Acumula as PARTES de cada prédio vindas de TODOS os arquivos do
        # diretório-folha (partições por ano/part-files) antes do corte.
        parts = defaultdict(list)
        data_files = sorted(f for f in leaf.iterdir() if f.is_file()
                            and f.suffix.lower() in (".parquet", ".csv"))
        # FASE A — leitura: barra por arquivo do grupo, com nº de prédios
        # já descobertos no postfix. Cronometrada + contagem de linhas lidas.
        t0_read, rows_read = time.time(), 0
        fbar = tqdm(data_files, desc=f"({i}/{len(leaves)}) ler {group_rel}",
                    unit="arquivo", file=sys.stdout, dynamic_ncols=True)
        for fp in fbar:
            for bname, bdf in iter_building_series_file(fp):
                parts[bname].append(bdf)
                rows_read += len(bdf)
            if hasattr(fbar, "set_postfix"):
                fbar.set_postfix(prédios=len(parts), linhas=f"{rows_read:,}")
        t_read = time.time() - t0_read

        # FASE B — split + gravação: barra por prédio (é aqui que grupos
        # grandes passam a maior parte do tempo: milhares de parquets
        # pequenos escritos em disco/NAS).
        t0_write, rows_written = time.time(), 0
        # LINHA CRUCIAL (--fine-resume): a ordem de escrita dos prédios é
        # DETERMINÍSTICA (arquivos ordenados + extração em ordem de
        # aparição), então, dos prédios com saída já existente, todos estão
        # completos EXCETO o último — que é reescrito por segurança.
        skip_done: set = set()
        if fine_resume:
            existing = [b for b in parts
                        if (CFG.split_root / "train" / group_rel /
                            f"{b}.parquet").exists()]
            skip_done = set(existing[:-1])       # último existente é refeito
            if existing:
                print(f"[step1][fine-resume] {group_rel}: "
                      f"{len(skip_done)} prédios já escritos serão pulados; "
                      f"reescrevendo a partir de '{existing[-1]}'", flush=True)

        t_w0 = time.time()
        rows_written = 0
        bbar = tqdm(parts.items(), total=len(parts),
                    desc=f"({i}/{len(leaves)}) gravar {group_rel}",
                    unit="prédio", file=sys.stdout, dynamic_ncols=True)
        for bname, chunks in bbar:
            if bname in skip_done:
                continue
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
                rows_written += len(part)
            n_ok += 1
        # LINHA CRUCIAL: o marcador só é gravado APÓS todos os parquets do
        # grupo estarem em disco — um kill no meio deixa o grupo sem
        # marcador e ele será refeito integralmente na próxima execução.
        t_write = time.time() - t0_write
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "buildings": len(parts), "rows_read": rows_read,
            "rows_written": rows_written,
            "t_read_s": round(t_read, 1), "t_write_s": round(t_write, 1)}))
        tot_rows_read += rows_read
        tot_rows_written += rows_written
        # Resumo do grupo: total | leitura (linhas) | escrita (linhas)
        print(f"[step1] ({i}/{len(leaves)}) {group_rel}: "
              f"{len(parts)} prédios | total {_fmt_dur(t_read + t_write)} | "
              f"leitura {_fmt_dur(t_read)} ({rows_read:,} linhas) | "
              f"escrita {_fmt_dur(t_write)} ({rows_written:,} linhas)",
              flush=True)
    print(f"[step1] prédios processados: {n_ok} | descartados (curtos): {n_skip}",
          flush=True)
    print(f"[step1] TEMPO TOTAL: {_fmt_dur(time.time() - t0_script)} | "
          f"linhas lidas: {tot_rows_read:,} | "
          f"linhas escritas: {tot_rows_written:,}", flush=True)
    print(f"[step1] saída em: {CFG.split_root}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adopt", action="store_true",
                    help="adota como concluídos os grupos que já têm saída em "
                         "01_splits (de execuções anteriores ao manifesto), "
                         "refazendo apenas o último grupo da ordem")
    ap.add_argument("--fine-resume", action="store_true",
                    help="dentro de um grupo interrompido, PULA os prédios já "
                         "escritos (reescrevendo só o último) em vez de refazer "
                         "o grupo inteiro; a leitura dos arquivos-fonte é "
                         "refeita de qualquer forma")
    args = ap.parse_args()
    run(adopt=args.adopt, fine_resume=args.fine_resume)
