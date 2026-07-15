"""
GRÁFICO 2 — Decomposição wavelet POR JANELA, lida DIRETAMENTE dos .pt
gerados pelo passo 2-3 (02_windows): o gráfico mostra o que REALMENTE
rodou (tensores salvos), sem recalcular nada.

Fluxo interativo (igual ao plot1):
  1. lista os grupos (Sector/Grupo) que possuem .pt no split escolhido;
  2. você escolhe o grupo;
  3. lista os prédios registrados DENTRO do .pt do grupo;
  4. você escolhe o prédio;
  5. gera DUAS figuras com eixo x compartilhado e a janela "andando"
     pelo stride, com assíntotas verticais:
       FIG A (--out):  decomposição armazenada (original, trend, sazonal);
       FIG B (--out2): banda de anomalia — trend deslocado para CIMA
         (+q99 do resíduo) e para BAIXO (+q01), a MESMA banda bicaudal
         intra-janela do passo 4; pontos fora da banda em vermelho.
         Sem componente sazonal; série original mais grossa na janela.

Uso:
    python plot2_decomposition.py                  # menus interativos
    python plot2_decomposition.py --group Residential/Prayas --building casa_A
    python plot2_decomposition.py --list           # inventário e sai
    (--split test para inspecionar o janelamento do conjunto de teste)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from config import CFG
from plot1_split import _menu

sns.set_theme(style="white", context="notebook")


# ---------------------------------------------------------------------------
# Navegação sobre a saída do passo 2-3 (02_windows)
# ---------------------------------------------------------------------------
def list_groups(split: str) -> list[str]:
    """Grupos 'Sector/Grupo' que já possuem .pt gerado no split."""
    base = CFG.windows_root / split
    if not base.exists():
        raise FileNotFoundError(
            f"Janelas não encontradas em {base} — rode o passo 2-3 antes.")
    return sorted(f"{s.name}/{pt.stem}"
                  for s in base.iterdir() if s.is_dir()
                  for pt in s.glob("*.pt"))


def load_pack(split: str, group: str) -> dict:
    pt = CFG.windows_root / split / f"{group}.pt"
    # LINHA CRUCIAL: carrega o ARTEFATO salvo pelo passo 2-3 — o gráfico
    # exibe exatamente os tensores que alimentarão o treino, e não uma
    # decomposição recalculada agora (que poderia divergir do que rodou).
    return torch.load(pt, weights_only=False)


def select_windows(pack: dict, building: str, skip: int, n: int):
    """Janelas armazenadas do prédio, em ordem temporal (por 'start')."""
    b_idx = pack["buildings"].index(building)
    mask = np.asarray(pack["building_idx"]) == b_idx
    starts = np.asarray(pack["start"])[mask]
    x = np.asarray(pack["x"])[mask]
    trend = np.asarray(pack["trend"])[mask]
    season = np.asarray(pack["season"])[mask]
    order = np.argsort(starts)
    sel = order[skip: skip + n]
    return starts[sel], x[sel], trend[sel], season[sel]


def background_series(split: str, group: str, building: str,
                      starts, x) -> np.ndarray:
    """Série de fundo: preferencialmente o parquet do split; se ausente,
    reconstrói costurando as próprias janelas salvas (sem recomputar)."""
    fp = CFG.split_root / split / group / f"{building}.parquet"
    if fp.exists():
        return np.nan_to_num(
            pd.read_parquet(fp)["energy"].to_numpy(dtype=float))
    W = x.shape[1]
    s = np.zeros(int(starts.max()) + W)
    for st, w in zip(starts, x):
        s[int(st): int(st) + W] = w
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train",
                    choices=list(CFG.splits))
    ap.add_argument("--group", type=str, default=None,
                    help="Sector/Grupo (ex.: Residential/Prayas); "
                         "sem isso, abre menu interativo")
    ap.add_argument("--building", type=str, default=None)
    ap.add_argument("--list", action="store_true",
                    help="apenas lista grupos/prédios com janelas geradas")
    ap.add_argument("--n", type=int, default=4, help="nº de janelas (subplots)")
    ap.add_argument("--skip", type=int, default=0,
                    help="pula as k primeiras janelas (escolher trecho)")
    ap.add_argument("--pad", type=int, default=24,
                    help="folga (pts) exibida antes/depois do conjunto de janelas")
    ap.add_argument("--out", type=Path, default=Path("fig2_decomposicao.png"))
    ap.add_argument("--out2", type=Path, default=Path("fig2_banda_anomalia.png"))
    args = ap.parse_args()

    groups = list_groups(args.split)
    if args.list:
        for g in groups:
            pack = load_pack(args.split, g)
            print(f"{g}: {len(pack['buildings'])} prédios, "
                  f"{len(pack['start'])} janelas")
        return

    group = args.group if args.group in groups else _menu(groups, "GRUPOS:")
    pack = load_pack(args.split, group)
    buildings = pack["buildings"]
    building = (args.building if args.building in buildings
                else _menu(sorted(buildings), f"PRÉDIOS de {group}:"))

    starts, x, trend, season = select_windows(pack, building,
                                              args.skip, args.n)
    if len(starts) == 0:
        raise RuntimeError(f"Nenhuma janela para {group}/{building} "
                           f"com skip={args.skip}")
    W = x.shape[1]
    B = pack["meta"]["backcast"]
    S = pack["meta"]["stride"]

    s = background_series(args.split, group, building, starts, x)
    lo = max(int(starts[0]) - args.pad, 0)
    hi = min(int(starts[-1]) + W + args.pad, len(s))
    t_bg = np.arange(lo, hi)

    def make_frames(n_sub):
        """Esqueleto comum às duas figuras: fundo, assíntotas e seta de stride."""
        fig, axes = plt.subplots(n_sub, 1, figsize=(12, 2.4 * n_sub),
                                 sharex=True, sharey=True)
        return fig, np.atleast_1d(axes)

    def frame_scaffold(ax, k, start):
        ax.plot(t_bg, s[lo:hi], color="0.75", lw=0.7)
        ax.axvline(start, color="k", ls="--", lw=1)
        ax.axvline(start + W, color="k", ls="--", lw=1)
        ax.axvline(start + B, color="crimson", ls=":", lw=1, alpha=0.8)
        if k > 0:
            y_arrow = ax.get_ylim()[0] + 0.08 * np.ptp(ax.get_ylim())
            ax.annotate("", xy=(start, y_arrow), xytext=(start - S, y_arrow),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
            ax.text(start - S / 2, y_arrow, f" stride={S}", fontsize=9,
                    color="crimson", va="bottom", ha="center")
        ax.set_ylabel("Energia", fontsize=9)
        ax.set_title(f"Janela {args.skip + k + 1}: t=[{start}, {start + W})",
                     fontsize=10, loc="left")

    # ------------------------- FIGURA A: decomposição -------------------------
    fig, axes = make_frames(len(starts))
    for k, ax in enumerate(axes):
        start = int(starts[k])
        frame_scaffold(ax, k, start)
        t = np.arange(start, start + W)
        ax.plot(t, x[k], color="0.35", lw=1.8, label="original")
        ax.plot(t, trend[k], color="#356e9c", lw=2.2, label="trend (wavelet)")
        ax.plot(t, season[k], color="seagreen", lw=0.9, label="sazonal/resíduo")

    axes[0].legend(ncol=3, fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Passo de tempo (h)")
    fig.suptitle(f"Decomposição por janela ({args.split}, janela={W}, "
                 f"stride={S}) — {group}/{building}", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"salvo em {args.out.resolve()}")

    # ------------------- FIGURA B: banda de anomalia (passo 4) ----------------
    fig, axes = make_frames(len(starts))
    for k, ax in enumerate(axes):
        start = int(starts[k])
        frame_scaffold(ax, k, start)
        t = np.arange(start, start + W)

        # LINHAS CRUCIAIS: MESMOS deslocamentos do label_window (passo 4) —
        # resíduo intra-janela e quantis bicaudais do config. A banda é
        # [trend + q01, trend + q99]; fora dela = anômalo.
        resid = x[k] - trend[k]
        thr_low = np.quantile(resid, CFG.anomaly_quantile_low)
        thr_high = np.quantile(resid, CFG.anomaly_quantile_high)
        lower, upper = trend[k] + thr_low, trend[k] + thr_high
        outside = (x[k] < lower) | (x[k] > upper)

        ax.fill_between(t, lower, upper, color="#356e9c", alpha=0.18,
                        label=(f"banda [q{CFG.anomaly_quantile_low:.0%}, "
                               f"q{CFG.anomaly_quantile_high:.0%}]"))
        ax.plot(t, upper, color="#356e9c", lw=1.0, ls="--")
        ax.plot(t, lower, color="#356e9c", lw=1.0, ls="--")
        ax.plot(t, trend[k], color="#356e9c", lw=2.2, label="trend (wavelet)")
        # série original MAIS GROSSA dentro da janela:
        ax.plot(t, x[k], color="0.25", lw=1.8, label="original")
        if outside.any():
            ax.scatter(t[outside], x[k][outside], s=22, color="red",
                       zorder=4, label="fora da banda (anômalo)")

    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_, ncol=2, fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Passo de tempo (h)")
    fig.suptitle(f"Banda de anomalia por janela ({args.split}, "
                 f"q=[{CFG.anomaly_quantile_low}, {CFG.anomaly_quantile_high}], "
                 f"janela={W}, stride={S}) — {group}/{building}", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out2, dpi=150)
    print(f"salvo em {args.out2.resolve()}")


if __name__ == "__main__":
    main()