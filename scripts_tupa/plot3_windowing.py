"""
GRÁFICO 3 — "Frames" (subplots) do janelamento sobre a série JÁ ROTULADA
pelo passo 4-5 (03_labeled_series): mostra o que REALMENTE rodou.

Em cada frame:
    - série de fundo com os PONTOS ANÔMALOS FUNDIDOS (coluna 'anomaly')
      marcados em vermelho;
    - faixa azul   = backcast (entrada do modelo);
    - faixa laranja= forecast/validação (val_horizon_steps), subsequente;
    - linhas verticais tracejadas = limites da janela;
    - seta anotando o stride entre frames consecutivos.
Janela/stride/backcast vêm do meta do .pt do passo 2-3 (o que foi
executado), com fallback para o config atual.

Fluxo interativo (igual ao plot1/plot2):
  1. lista os grupos (Sector/Grupo) com rótulos gerados no split;
  2. você escolhe o grupo; 3. lista os prédios; 4. você escolhe o prédio.

Uso:
    python plot3_windowing.py                      # menus interativos
    python plot3_windowing.py --group Residential/Prayas --building casa_A
    python plot3_windowing.py --list               # inventário e sai
    (--no-labels para ocultar as marcações de anomalia)
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
# Navegação sobre a saída do passo 4-5 (03_labeled_series)
# ---------------------------------------------------------------------------
def list_groups(split: str) -> list[str]:
    """Grupos 'Sector/Grupo' que já possuem séries rotuladas no split."""
    base = CFG.labels_root / split
    if not base.exists():
        raise FileNotFoundError(
            f"Rótulos não encontrados em {base} — rode o passo 4-5 antes.")
    return sorted(f"{s.name}/{g.name}"
                  for s in base.iterdir() if s.is_dir()
                  for g in s.iterdir() if g.is_dir())


def list_buildings(split: str, group: str) -> list[str]:
    return sorted(p.stem for p in
                  (CFG.labels_root / split / group).glob("*.parquet"))


def load_labeled_series(split: str, group: str, building: str):
    """Série + rótulos fundidos gravados pelo passo 4-5 (sem recomputar)."""
    fp = CFG.labels_root / split / group / f"{building}.parquet"
    # LINHA CRUCIAL: lê o ARTEFATO rotulado — a coluna 'anomaly' exibida é
    # exatamente a que o passo 4-5 fundiu e gravou, não um recálculo.
    df = pd.read_parquet(fp)
    y = np.nan_to_num(df["energy"].to_numpy(dtype=float))
    lab = (df["anomaly"].to_numpy(dtype=int)
           if "anomaly" in df.columns else np.zeros(len(y), dtype=int))
    return y, lab


def window_meta(split: str, group: str) -> tuple[int, int, int]:
    """(backcast, janela_total, stride) do meta do .pt; fallback: config."""
    pt = CFG.windows_root / split / f"{group}.pt"
    if pt.exists():
        meta = torch.load(pt, weights_only=False)["meta"]
        return (meta["backcast"],
                meta["backcast"] + meta["forecast"], meta["stride"])
    return (CFG.backcast_length,
            CFG.backcast_length + CFG.forecast_length, CFG.stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train",
                    choices=list(CFG.splits))
    ap.add_argument("--group", type=str, default=None,
                    help="Sector/Grupo (ex.: Residential/Prayas); "
                         "sem isso, abre menu interativo")
    ap.add_argument("--building", type=str, default=None)
    ap.add_argument("--list", action="store_true",
                    help="apenas lista grupos/prédios com rótulos gerados")
    ap.add_argument("--frames", type=int, default=5, help="nº de frames/subplots")
    ap.add_argument("--start", type=int, default=0, help="início do 1º frame")
    ap.add_argument("--span", type=int, default=None,
                    help="trecho da série exibido (padrão: janela + frames*stride + 2 dias)")
    ap.add_argument("--no-labels", action="store_true",
                    help="oculta as marcações de anomalia")
    ap.add_argument("--out", type=Path, default=Path("fig3_janelamento.png"))
    args = ap.parse_args()

    groups = list_groups(args.split)
    if args.list:
        for g in groups:
            print(f"{g}: {len(list_buildings(args.split, g))} prédios rotulados")
        return

    group = args.group if args.group in groups else _menu(groups, "GRUPOS:")
    buildings = list_buildings(args.split, group)
    building = (args.building if args.building in buildings
                else _menu(buildings, f"PRÉDIOS de {group}:"))

    B, W, S = window_meta(args.split, group)
    V = CFG.val_horizon_steps
    span = args.span or (W + args.frames * S + 2 * 24)

    full, lab_full = load_labeled_series(args.split, group, building)
    s = full[args.start: args.start + span]
    lab = lab_full[args.start: args.start + span]
    t = np.arange(args.start, args.start + len(s))
    rate = lab_full.mean()

    fig, axes = plt.subplots(args.frames, 1,
                             figsize=(12, 2.2 * args.frames),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        # Posição da janela no frame k = início + k*stride (regra do pipeline)
        w0 = args.start + k * S
        ax.plot(t, s, color="0.6", lw=0.8)
        if not args.no_labels and lab.any():
            # LINHA CRUCIAL: marca os pontos com anomaly==1 do passo 4-5 —
            # inspeção visual direta da rotulagem fundida que rodou.
            ax.scatter(t[lab == 1], s[lab == 1], s=14, color="red",
                       zorder=3, label="anomalia (fundida)" if k == 0 else None)

        ax.axvspan(w0, w0 + B, color="#356e9c", alpha=0.25)
        ax.axvspan(w0 + B, w0 + B + V, color="darkorange", alpha=0.35)
        ax.axvline(w0, color="k", ls="--", lw=1)
        ax.axvline(w0 + W, color="k", ls="--", lw=1)

        ax.text(w0 + B / 2, ax.get_ylim()[1] * 0.9, f"janela ({B})",
                ha="center", fontsize=9, color="#1f4e79", fontweight="bold")
        ax.text(w0 + B + V / 2, ax.get_ylim()[1] * 0.9, f"val ({V})",
                ha="center", fontsize=9, color="#b35900", fontweight="bold")

        if k > 0:
            y_arrow = ax.get_ylim()[0] + 0.08 * np.ptp(ax.get_ylim())
            ax.annotate("", xy=(w0, y_arrow), xytext=(w0 - S, y_arrow),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
            ax.text(w0 - S / 2, y_arrow, f" stride={S}", fontsize=9,
                    color="crimson", va="bottom", ha="center")
        ax.set_ylabel(f"frame {k+1}", fontsize=9)

    if not args.no_labels and lab.any():
        axes[0].legend(fontsize=9, loc="upper right")
    axes[-1].set_xlabel("Passo de tempo (h)")
    fig.suptitle(f"Janelamento + rótulos ({args.split}, janela={B}, stride={S}, "
                 f"validação={V}, taxa anomalia={rate:.2%}) — "
                 f"{group}/{building}", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"salvo em {args.out.resolve()}")


if __name__ == "__main__":
    main()