"""
Configuração central do pipeline de detecção de anomalias.
Todos os passos (1 a 8) leem deste arquivo para garantir consistência
— em especial, os MESMOS parâmetros de janelamento, wavelet e rotulagem
são usados no treino (passos 2-5) e no teste em tempo de execução (passo 8).
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # ---------- Caminhos ----------
    raw_root: Path = Path(r"C:\Users\paula\source\EnergyBench\Dataset_V0.0\Energy-Load-Profiles")
    resolution: str = "Hourly"                    # "15min" | "30min" | "Hourly"
    out_root: Path = Path(r"D:\Juliana\Dataset-diss\EnergyBench-Anomaly")

    # ---------- Passo 1: split ----------
    split_mode: str = "temporal"                  # "temporal" (por edifício, no tempo)
                                                  # ou "by_building" (edifícios inteiros por split)
    train_frac: float = 0.85                      # teste = 1 - train (SEM val estático)
    min_series_len: int = 24 * 30                 # descarta séries com < 30 dias

    # ---------- Passo 2: janelamento ----------
    backcast_length: int = 168                    # 7 dias (horário) — igual ao W-LSTMix
    forecast_length: int = 24
    stride: int = 24                              # sobreposição: janelas deslizam 1 dia
    period: int = 24

    # ---------- Passo 3: wavelet ----------
    wavelet: str = "db4"
    wavelet_level: int = 5                        # será truncado por dwt_max_level se preciso

    # ---------- Passos 4-5: rótulos ----------
    # Limiar BICAUDAL: anômalo se resíduo < q_low OU resíduo > q_high
    anomaly_quantile_low: float = 0.01
    anomaly_quantile_high: float = 0.99
    merge_rule: str = "any"                       # "any" (união) | "majority" (voto)

    # ---------- Passos 6-7: treino com validação rolling-origin ----------
    # Comportamento da figura: treino expande; validação = 1 dia subsequente.
    val_horizon_steps: int = 24                   # tamanho da validação (1 dia, horário)
    n_rolling_folds: int = 4                      # nº de origens (folds) da figura
    initial_train_frac: float = 0.80              # fração do split de treino usada
                                                  # como treino no fold 0
    rolling_mode: str = "expanding"               # "expanding" (figura) | "sliding"
    train_span_steps: int = 24 * 90               # tamanho fixo do treino se "sliding"
    warm_start: bool = True                       # continua o mesmo modelo entre folds
    epochs_per_fold: int = 10                     # épocas máximas por fold
    batch_size: int = 512
    patience: int = 3                             # early stopping por fold
    learning_rate: float = 1e-3
    lambda_cls: float = 1.0                       # peso da perda de classificação
    seed: int = 42

    # Hiperparâmetros do backbone (espelham configs/W_LSTMix.json do repo original)
    num_blocks_per_stack: int = 1
    patch_size: int = 24
    thetas_dim: int = 32
    hidden_dim: int = 64
    embed_dim: int = 32
    num_heads: int = 4
    ff_hidden_dim: int = 64

    splits: tuple = ("train", "test")
    sectors: tuple = ("Commercial", "Residential")

    # ---------- Derivados ----------
    @property
    def split_root(self) -> Path:
        return self.out_root / "01_splits" / self.resolution

    @property
    def windows_root(self) -> Path:
        return self.out_root / "02_windows" / self.resolution

    @property
    def labels_root(self) -> Path:
        return self.out_root / "03_labeled_series" / self.resolution

    @property
    def models_root(self) -> Path:
        return self.out_root / "04_models"


CFG = PipelineConfig()
