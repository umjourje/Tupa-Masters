"""
GRÁFICO 1 — Série temporal de UM prédio com a partição treino/teste,
lida DIRETAMENTE da saída do passo 1 (01_splits).

Fluxo interativo:
  1. lista os grupos disponíveis (Sector/Grupo) encontrados no split;
  2. você escolhe o grupo;
  3. lista os prédios do grupo (presentes em train);
  4. você escolhe o prédio;
  5. plota treino+teste concatenados, com a assíntota vertical exatamente
     na fronteira REAL do split (fim do arquivo de treino) — não em uma
     fração recalculada.

Uso:
    python plot1_split.py                          # menus interativos
    python plot1_split.py --group Residential/Prayas --building casa_A
    python plot1_split.py --list                   # só listar e sair
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CFG

DEFAULT_FILE = CFG.raw_root / CFG.resolution / "Residential" / "Prayas"  # legado (plot2/3)

sns.set_theme(style="whitegrid", context="talk")


def pick_file(path: Path) -> Path:
    """(Legado, usado por load_building_series/plot2/plot3.)"""
    if path.is_file():
        return path
    cands = sorted(list(path.glob("*.parquet")) + list(path.glob("*.csv")))
    if not cands:
        raise FileNotFoundError(f"Nenhum parquet/csv em {path}")
    return cands[0]


def load_building_series(path: Path, building: str | None = None):
    """(Legado, usado por plot2/plot3 sobre arquivos brutos.)"""
    from step1_split import iter_building_series
    fp = pick_file(path)
    df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
    chosen = None
    for bname, bdf in iter_building_series(df, fp.stem):
        if building is None or bname == building:
            chosen = (bname, bdf)
            break
    if chosen is None:
        raise ValueError(f"Prédio '{building}' não encontrado em {fp}")
    bname, bdf = chosen
    if "timestamp" in bdf.columns:
        bdf = bdf.sort_values("timestamp")
    y = np.nan_to_num(bdf["energy"].to_numpy(dtype=float))
    return f"{fp.stem}:{bname}", y


# ---------------------------------------------------------------------------
# Navegação sobre a saída do passo 1 (01_splits)
# ---------------------------------------------------------------------------
def list_groups() -> list[str]:
    """Grupos disponíveis, como 'Sector/Grupo' (baseado no split de treino)."""
    base = CFG.split_root / "train"
    if not base.exists():
        raise FileNotFoundError(
            f"Split não encontrado em {base} — rode step1_split.py antes.")
    return sorted(f"{s.name}/{g.name}"
                  for s in base.iterdir() if s.is_dir()
                  for g in s.iterdir() if g.is_dir())


def list_buildings(group: str) -> list[str]:
    """Prédios do grupo (nomes dos parquets no split de treino)."""
    return sorted(p.stem for p in
                  (CFG.split_root / "train" / group).glob("*.parquet"))


def _menu(options: list[str], titulo: str) -> str:
    print(f"\n{titulo}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i:>3}] {opt}")
    while True:
        raw = input("Escolha o número (ou nome exato): ").strip()
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Opção inválida, tente novamente.")


def load_split_series(group: str, building: str):
    """Lê treino e teste do prédio e devolve (y_concat, fronteira_real)."""
    ys = {}
    for split in ("train", "test"):
        fp = CFG.split_root / split / group / f"{building}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            ys[split] = np.nan_to_num(df["energy"].to_numpy(dtype=float))
        else:
            ys[split] = np.array([])
    # LINHA CRUCIAL: a assíntota fica na fronteira REAL gravada pelo passo 1
    # (comprimento do arquivo de treino), e não em uma fração recalculada —
    # se CFG.train_frac mudar depois do split, o gráfico continua fiel.
    boundary = len(ys["train"])
    return np.concatenate([ys["train"], ys["test"]]), boundary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, default=None,
                    help="Sector/Grupo (ex.: Residential/Prayas); "
                         "sem isso, abre menu interativo")
    ap.add_argument("--building", type=str, default=None,
                    help="nome do prédio dentro do grupo")
    ap.add_argument("--list", action="store_true",
                    help="apenas lista grupos e prédios disponíveis")
    ap.add_argument("--out", type=Path, default=Path("fig1_split.png"))
    args = ap.parse_args()

    groups = list_groups()
    if args.list:
        for g in groups:
            print(f"{g}: {len(list_buildings(g))} prédios")
        return

    group = args.group if args.group in groups else _menu(groups, "GRUPOS:")
    buildings = list_buildings(group)
    building = (args.building if args.building in buildings
                else _menu(buildings, f"PRÉDIOS de {group}:"))

    y, i_tr = load_split_series(group, building)
    if len(y) == 0:
        raise RuntimeError(f"Série vazia para {group}/{building}")
    x = range(len(y))
    frac_tr = i_tr / len(y)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y, lw=0.6, color="#356e9c")
    ax.axvline(i_tr, color="crimson", ls="--", lw=2)
    ax.axvspan(0, i_tr, color="#356e9c", alpha=0.08)
    ax.axvspan(i_tr, len(y), color="darkorange", alpha=0.10)
    ymax = ax.get_ylim()[1]
    ax.text(i_tr / 2, ymax * 0.95, f"TREINO ({frac_tr:.0%})",
            ha="center", fontweight="bold", color="#356e9c")
    ax.text((i_tr + len(y)) / 2, ymax * 0.95, f"TESTE ({1 - frac_tr:.0%})",
            ha="center", fontweight="bold", color="darkorange")

    ax.set_xlabel("Passo de tempo (h)")
    ax.set_ylabel("Energia")
    ax.set_title(f"Partição treino/teste — {group}/{building}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"salvo em {args.out.resolve()}")


if __name__ == "__main__":
    main()