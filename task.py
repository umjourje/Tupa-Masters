"""task.py — Lógica de modelo, dados, treino e avaliação do W-LSTMix para uso federado (Flower).

Este módulo adapta o código centralizado do repositório W-LSTMix
(EdgeIntelligenceLab/W-LSTMix, licença MIT, ICML Workshop FMSD 2025)
para ser consumido pelo client_app.py e pelo server_app.py de um
Flower App (Message API, Flower >= 1.21).

Diferenças em relação ao train.py/test.py originais (centralizados):
  (a) O laço de treino virou uma função pura que RETORNA métricas
      (sem tqdm, sem print como mecanismo principal de registro);
  (b) O early stopping local e o checkpoint local (best_model.pth)
      foram REMOVIDOS do cliente — no cenário federado, a decisão de
      "melhor modelo" pertence ao servidor (estratégia), que compara
      métricas agregadas por rodada;
  (c) As métricas do test.py (CVRMSE, NRMSE, MAE, MSE) são calculadas
      localmente e retornadas como floats — prontas para viajar num
      MetricRecord até o servidor;
  (d) O carregamento de dados é parametrizado por partition_id, lido
      de context.node_config no client_app.py (substitui o antigo
      argumento --node-id).

Estrutura de diretórios esperada em CADA dispositivo (Raspberry):

    <data_root>/
        train/  <região>/<prédio>.csv|.parquet   (coluna 'energy')
        val/    <região>/<prédio>.csv|.parquet   (opcional)
        test/   <região>/<prédio>.csv|.parquet

Também são aceitos arquivos soltos diretamente em train/, val/, test/.

Dependências copiadas do repositório original (mesma pasta do projeto):
    models/W_LSTMix.py        -> classe Model
    my_utils/metrics.py       -> cal_cvrmse, cal_nrmse, cal_mae, cal_mse
    configs/W_LSTMix.json     -> hiperparâmetros do modelo

ATENÇÃO: as fórmulas exatas de CVRMSE/NRMSE do paper estão em
my_utils/metrics.py. Este módulo IMPORTA essas funções para garantir
comparabilidade com os resultados centralizados. O fallback inline
(abaixo) usa as fórmulas usuais da literatura e só é acionado se o
import falhar — nesse caso, um aviso é registrado e a comparabilidade
com o paper deve ser conferida manualmente.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pywt
import torch
from statsmodels.tsa.seasonal import seasonal_decompose
from torch.utils.data import ConcatDataset, DataLoader, Dataset

log = logging.getLogger("wlstmix.task")

# ---------------------------------------------------------------------------
# Imports do repositório W-LSTMix (copiar as pastas models/ e my_utils/)
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.dirname(__file__))

from models import W_LSTMix  # noqa: E402  (modelo original, inalterado)

try:
    from my_utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse
    _METRICS_FROM_PAPER = True
except ImportError:  # Fallback com fórmulas usuais — verificar comparabilidade!
    _METRICS_FROM_PAPER = False
    log.warning(
        "my_utils/metrics.py não encontrado; usando fórmulas padrão de "
        "CVRMSE/NRMSE/MAE/MSE. Confira a equivalência com o paper."
    )

    def cal_mse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean((pred - true) ** 2))

    def cal_mae(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - true)))

    def cal_cvrmse(pred: np.ndarray, true: np.ndarray) -> float:
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        return rmse / (float(np.mean(true)) + 1e-8)

    def cal_nrmse(pred: np.ndarray, true: np.ndarray) -> float:
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        return rmse / (float(np.max(true) - np.min(true)) + 1e-8)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "W_LSTMix.json")


def load_config(config_path: str = _DEFAULT_CONFIG) -> dict:
    """Carrega o JSON de configuração original do W-LSTMix."""
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pré-processamento (copiado do train.py original, sem alterações de lógica)
# ---------------------------------------------------------------------------
def standardize_series(series: np.ndarray, eps: float = 1e-8):
    mean = np.mean(series)
    std = np.std(series)
    return (series - mean) / (std + eps), mean, std


def unscale_predictions(predictions: np.ndarray, mean: float, std: float,
                        eps: float = 1e-8) -> np.ndarray:
    return predictions * (std + eps) + mean


def decompose_series(series: np.ndarray, method_decom: str, period: int = 24,
                     wavelet: str = "db4", level: Optional[int] = 5):
    """Decompõe a série em tendência e sazonalidade+resíduo (idem original)."""
    if method_decom == "seasonal_decompose":
        result = seasonal_decompose(
            series, model="additive", period=period, extrapolate_trend="freq"
        )
        trend = result.trend
        seasonal_plus_resid = series - trend
        trend = pd.Series(trend).bfill().ffill().values
        seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values
        return trend, seasonal_plus_resid

    if method_decom == "wavelet":
        if level is None:
            level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec(series, wavelet, level=level)
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        trend = pywt.waverec(trend_coeffs, wavelet)[: len(series)]
        seasonal_plus_resid = series - trend
        seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values
        return trend, seasonal_plus_resid

    raise ValueError(f"method_decom desconhecido: {method_decom!r}")


class DecomposedTimeSeriesDataset(Dataset):
    """Dataset de janelas (backcast/forecast) sobre série decomposta (idem original)."""

    def __init__(self, series, backcast_length, forecast_length, method_decom,
                 stride: int = 1, period: int = 24):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride
        self.method_decom = method_decom

        trend, seasonality = decompose_series(series, method_decom, period=period)
        self.trend, self.trend_mean, self.trend_std = standardize_series(trend)
        self.season, self.season_mean, self.season_std = standardize_series(seasonality)

    def __len__(self):
        return (len(self.trend) - self.backcast_length - self.forecast_length) \
            // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        bl, fl = self.backcast_length, self.forecast_length
        return {
            "trend_input": torch.tensor(self.trend[start: start + bl],
                                        dtype=torch.float32),
            "season_input": torch.tensor(self.season[start: start + bl],
                                         dtype=torch.float32),
            "trend_target": torch.tensor(self.trend[start + bl: start + bl + fl],
                                         dtype=torch.float32),
            "season_target": torch.tensor(self.season[start + bl: start + bl + fl],
                                          dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Carregamento de dados por partição (adaptação federada do load_datasets)
# ---------------------------------------------------------------------------
def _iter_series_files(folder_path: str):
    """Percorre <pasta>/<região>/<arquivo> e também arquivos soltos em <pasta>."""
    for entry in sorted(os.listdir(folder_path)):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            for fname in sorted(os.listdir(entry_path)):
                yield entry, os.path.join(entry_path, fname)
        else:
            yield "_root", entry_path


def _read_energy(file_path: str) -> Optional[np.ndarray]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        return None
    if "energy" not in df.columns:
        log.warning("Coluna 'energy' ausente em %s — arquivo ignorado.", file_path)
        return None
    return df["energy"].values


def load_split(data_root: str, split: str, cfg: dict) -> Optional[ConcatDataset]:
    """Carrega um split ('train' | 'val' | 'test') do diretório local do cliente.

    No cenário federado em hardware físico, cada Raspberry já contém apenas
    a SUA partição de dados em <data_root>; por isso, diferentemente da
    simulação, não há particionador — a partição É o disco local.
    Retorna None se o split não existir (ex.: cliente sem conjunto de validação).
    """
    folder = os.path.join(data_root, split)
    if not os.path.isdir(folder):
        return None
    datasets = []
    for _, file_path in _iter_series_files(folder):
        series = _read_energy(file_path)
        if series is None:
            continue
        datasets.append(
            DecomposedTimeSeriesDataset(
                series,
                cfg["backcast_length"],
                cfg["forecast_length"],
                cfg["method_decom"],
                cfg.get("stride", 1),
                cfg.get("period", 24),
            )
        )
    if not datasets:
        return None
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def get_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Instancia o W-LSTMix com a MESMA assinatura usada no train.py original.

    IMPORTANTE (federação): todos os clientes e o servidor devem usar o mesmo
    configs/W_LSTMix.json, pois o FedAvg pressupõe arquiteturas idênticas.
    """
    return W_LSTMix.Model(
        device=device,
        num_blocks_per_stack=cfg["num_blocks_per_stack"],
        forecast_length=cfg["forecast_length"],
        backcast_length=cfg["backcast_length"],
        patch_size=cfg["patch_size"],
        num_patches=cfg["backcast_length"] // cfg["patch_size"],
        thetas_dim=cfg["thetas_dim"],
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ff_hidden_dim=cfg["ff_hidden_dim"],
    ).to(device)


def get_criterion(cfg: dict) -> torch.nn.Module:
    if cfg.get("loss", "mse") == "mse":
        return torch.nn.MSELoss()
    return torch.nn.HuberLoss(reduction="mean", delta=1)


def _weighted_loss(criterion, trend_pred, trend_target, season_pred, season_target):
    """Ponderação dinâmica alpha/beta idêntica à do repositório original."""
    loss_trend = criterion(trend_pred, trend_target)
    loss_season = criterion(season_pred, season_target)
    sum_loss = loss_trend + loss_season
    alpha = loss_season / sum_loss
    beta = loss_trend / sum_loss
    return alpha * loss_trend + beta * loss_season


# ---------------------------------------------------------------------------
# Treino local (adaptação do train() original — ver notas (a) e (b) no topo)
# ---------------------------------------------------------------------------
def train(model: torch.nn.Module, train_dataset, cfg: dict, device: torch.device,
          local_epochs: int, val_dataset=None) -> dict:
    """Treina o modelo por `local_epochs` épocas na partição local.

    Retorna um dicionário APENAS com valores numéricos (floats/ints/listas de
    floats), pronto para ser colocado num flwr MetricRecord:
        train_loss        : loss médio da ÚLTIMA época local
        train_loss_epochs : lista do loss médio por época (curva local)
        val_loss          : loss médio de validação (ou ausente, se sem val/)
        num-examples      : nº de janelas de treino (peso do FedAvg)
    """
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True)

    model.train()
    epoch_losses: list[float] = []
    for epoch in range(local_epochs):
        batch_losses = []
        for batch in train_loader:
            trend_input = batch["trend_input"].to(device)
            season_input = batch["season_input"].to(device)
            trend_target = batch["trend_target"].to(device)
            season_target = batch["season_target"].to(device)

            optimizer.zero_grad()
            trend_pred, season_pred = model(trend_input, season_input)
            total_loss = _weighted_loss(criterion, trend_pred, trend_target,
                                        season_pred, season_target)
            total_loss.backward()
            optimizer.step()
            batch_losses.append(total_loss.item())

        avg = float(np.mean(batch_losses))
        epoch_losses.append(avg)
        log.info("Época local %d/%d — train_loss=%.6f", epoch + 1, local_epochs, avg)

    metrics: dict = {
        "train_loss": epoch_losses[-1],
        "train_loss_epochs": epoch_losses,
        "num-examples": len(train_dataset),
    }

    # Validação local opcional (informativa; a decisão de "melhor modelo
    # global" é do servidor — ver nota (b) no topo deste arquivo)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                trend_pred, season_pred = model(
                    batch["trend_input"].to(device),
                    batch["season_input"].to(device),
                )
                val_losses.append(
                    _weighted_loss(
                        criterion, trend_pred,
                        batch["trend_target"].to(device),
                        season_pred,
                        batch["season_target"].to(device),
                    ).item()
                )
        metrics["val_loss"] = float(np.mean(val_losses))

    return metrics


# ---------------------------------------------------------------------------
# Avaliação local (adaptação do test.py — ver nota (c) no topo)
# ---------------------------------------------------------------------------
def evaluate(model: torch.nn.Module, cfg: dict, device: torch.device,
             data_root: str) -> dict:
    """Avalia o modelo global na partição de teste local do cliente.

    Replica o cálculo do test.py original (CVRMSE, NRMSE, MAE, MSE nas
    escalas desnormalizada e normalizada), porém agregando POR PARTIÇÃO
    (todos os prédios/arquivos deste cliente juntos) em vez de por prédio,
    e retornando floats para o MetricRecord. As médias/std de desnormalização
    são as de CADA dataset individual, como no original.
    """
    criterion = get_criterion(cfg)
    test_folder = os.path.join(data_root, "test")
    if not os.path.isdir(test_folder):
        raise FileNotFoundError(f"Split de teste não encontrado: {test_folder}")

    model.eval()
    losses: list[float] = []
    true_unscaled, pred_unscaled = [], []
    true_norm, pred_norm = [], []
    n_windows = 0

    for _, file_path in _iter_series_files(test_folder):
        series = _read_energy(file_path)
        if series is None:
            continue
        ds = DecomposedTimeSeriesDataset(
            series, cfg["backcast_length"], cfg["forecast_length"],
            cfg["method_decom"], cfg.get("stride", 1), cfg.get("period", 24),
        )
        loader = DataLoader(ds, batch_size=cfg["batch_size"])
        yt_t, yt_s, yp_t, yp_s = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                trend_pred, season_pred = model(
                    batch["trend_input"].to(device),
                    batch["season_input"].to(device),
                )
                losses.append(
                    _weighted_loss(
                        criterion, trend_pred,
                        batch["trend_target"].to(device),
                        season_pred,
                        batch["season_target"].to(device),
                    ).item()
                )
                yt_t.append(batch["trend_target"].numpy())
                yt_s.append(batch["season_target"].numpy())
                yp_t.append(trend_pred.cpu().numpy())
                yp_s.append(season_pred.cpu().numpy())

        yt_t = np.concatenate(yt_t, axis=0)
        yt_s = np.concatenate(yt_s, axis=0)
        yp_t = np.concatenate(yp_t, axis=0)
        yp_s = np.concatenate(yp_s, axis=0)
        n_windows += len(ds)

        # Escala normalizada (como armazenado no dataset)
        true_norm.append(yt_t + yt_s)
        pred_norm.append(yp_t + yp_s)

        # Escala original (desnormaliza cada componente com stats do dataset)
        true_unscaled.append(
            unscale_predictions(yt_t, ds.trend_mean, ds.trend_std)
            + unscale_predictions(yt_s, ds.season_mean, ds.season_std)
        )
        pred_unscaled.append(
            unscale_predictions(yp_t, ds.trend_mean, ds.trend_std)
            + unscale_predictions(yp_s, ds.season_mean, ds.season_std)
        )

    if n_windows == 0:
        raise RuntimeError(f"Nenhuma série válida em {test_folder}")

    y_true = np.concatenate(true_unscaled, axis=0)
    y_pred = np.concatenate(pred_unscaled, axis=0)
    y_true_n = np.concatenate(true_norm, axis=0)
    y_pred_n = np.concatenate(pred_norm, axis=0)

    return {
        "test_loss": float(np.mean(losses)),
        "cvrmse": float(cal_cvrmse(y_pred, y_true)),
        "nrmse": float(cal_nrmse(y_pred, y_true)),
        "mae": float(cal_mae(y_pred, y_true)),
        "mse": float(cal_mse(y_pred, y_true)),
        "mae_norm": float(cal_mae(y_pred_n, y_true_n)),
        "mse_norm": float(cal_mse(y_pred_n, y_true_n)),
        "metrics_from_paper_code": int(_METRICS_FROM_PAPER),
        "num-examples": n_windows,
    }
