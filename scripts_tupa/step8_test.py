"""
PASSO 8 — Teste com rótulos gerados EM TEMPO DE EXECUÇÃO.

O conjunto de teste não carrega coluna de rótulos. Aqui, para cada série
de teste, reexecutamos EXATAMENTE os passos 3 (wavelet por janela),
4 (percentil 99 por janela) e 5 (fusão) — importando as MESMAS funções
usadas no pré-processamento de treino — e comparamos as predições do
modelo com esses rótulos para produzir métricas (acurácia, precisão,
recall, F1, AUROC).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from config import CFG
from model_hybrid import HybridWLSTMix
from step2_3_windows_wavelet import iter_windows, decompose_window, standardize
from step4_5_labels import label_window, fuse_labels


@torch.no_grad()
def evaluate_series(model, device, series: np.ndarray):
    """Retorna (y_true_fundido, y_prob_fundido) por timestep da série."""
    B, F = CFG.backcast_length, CFG.forecast_length
    starts, win_labels = [], []
    prob_sum = np.zeros(len(series))
    prob_cnt = np.zeros(len(series))

    batch_ti, batch_si, batch_meta = [], [], []
    for start, w in iter_windows(series):
        # Passos 3+4 em runtime — mesmas funções do pré-processamento:
        trend, season = decompose_window(w)
        win_labels.append(label_window(w, trend))
        starts.append(start)
        t_n, *_ = standardize(trend)
        s_n, *_ = standardize(season)
        batch_ti.append(t_n[:B]); batch_si.append(s_n[:B])
        batch_meta.append(start)

    if not starts:
        return None, None

    ti = torch.tensor(np.stack(batch_ti)).to(device)
    si = torch.tensor(np.stack(batch_si)).to(device)
    probs = []
    for i in range(0, len(ti), CFG.batch_size):
        _, _, logits = model(ti[i:i+CFG.batch_size], si[i:i+CFG.batch_size])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)                 # (N_janelas, F)

    # Fusão das PROBABILIDADES por média nas sobreposições (horizonte F):
    for j, st in enumerate(batch_meta):
        prob_sum[st + B: st + B + F] += probs[j]
        prob_cnt[st + B: st + B + F] += 1

    # Passo 5 em runtime: fusão dos rótulos "verdade" gerados agora.
    y_true = fuse_labels(len(series), starts, np.stack(win_labels))

    covered = prob_cnt > 0                        # só avalia timesteps previstos
    y_prob = np.zeros(len(series))
    y_prob[covered] = prob_sum[covered] / prob_cnt[covered]
    return y_true[covered], y_prob[covered]


def run(model_path=None, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridWLSTMix(device).to(device)
    model_path = model_path or (CFG.models_root / "full" / "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    Y, P = [], []
    for sector in CFG.sectors:
        base = CFG.split_root / "test" / sector
        if not base.exists():
            continue
        for fp in sorted(base.rglob("*.parquet")):
            s = np.nan_to_num(pd.read_parquet(fp)["energy"]
                              .to_numpy(dtype=np.float64))
            yt, yp = evaluate_series(model, device, s)
            if yt is not None:
                Y.append(yt); P.append(yp)

    y = np.concatenate(Y); p = np.concatenate(P)
    pred = (p >= threshold).astype(int)
    metrics = {
        "accuracy":  float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "auroc":     float(roc_auc_score(y, p)) if y.min() != y.max() else None,
        "anomaly_rate_true": float(y.mean()),
        "n_points": int(len(y)),
    }
    (CFG.models_root / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[step8]", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
