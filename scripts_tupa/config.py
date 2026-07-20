"""
Configuração central do pipeline de detecção de anomalias.
Todos os passos (1 a 8) leem deste arquivo para garantir consistência
— em especial, os MESMOS parâmetros de janelamento, wavelet e rotulagem
são usados no treino (passos 2-5) e no teste em tempo de execução (passo 8).

Caminhos críticos vêm de variáveis de ambiente (arquivo .env na raiz do
projeto ou exportadas no shell), para não expor pastas de sistema no código:

    RAW_ROOT=/caminho/completo/ate/.../Energy-Load-Profiles
    OUT_ROOT=/caminho/completo/ate/a/pasta/de/saida

RAW_ROOT deve apontar DIRETAMENTE para a pasta que contém as resoluções
(15min/30min/Hourly) — ex.: .../Energy-Load-Profiles ou
.../Synthetic-Energy-Load-Profiles. Nada é concatenado depois.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os

try:                                    # dotenv é opcional: sem o pacote,
    from dotenv import load_dotenv      # vale o que estiver exportado no shell
    load_dotenv()
except ImportError:
    pass


def _get_path_env(var_name: str) -> Path:
    """Lê uma variável de ambiente obrigatória e converte para Path."""
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Variável de ambiente {var_name} não definida. "
            f"Defina-a no .env ou exporte no shell (caminho completo).")
    return Path(val)


@dataclass
class PipelineConfig:
    # ---------- Caminhos (obrigatórios, via .env) ----------
    raw_root: Path = field(default_factory=lambda: _get_path_env("RAW_ROOT"))
    out_root: Path = field(default_factory=lambda: _get_path_env("OUT_ROOT"))
    resolution: str = "Hourly"                    # "15min" | "30min" | "Hourly"

    # ---------- Passo 1: split ----------
    split_mode: str = "temporal"                  # "temporal" (por edifício, no tempo)
                                                  # ou "by_building" (edifícios inteiros por split)
    train_frac: float = 0.85                      # teste = 1 - train (SEM val estático)
    min_series_len: int = 24 * 30                 # descarta séries com < 30 dias
    # Orçamento de RAM (GB) para a LEITURA de arquivos wide no passo 1:
    # se o arquivo couber, é lido em UMA chamada (pyarrow paraleliza a
    # descompressão entre núcleos); senão, em lotes de colunas do maior
    # tamanho que caiba. Com 128 GB de RAM, 64 é um valor seguro.
    read_ram_budget_gb: float = 64.0
    # ---------- DuckDB (leituras/escritas massivas nos passos 2-5) ----------
    duckdb_threads: int = 0                       # 0 = automático (todos os núcleos)
    duckdb_memory_limit_gb: float = 64.0
    duckdb_files_per_batch: int = 8192            # parquets de prédio por query
    # ---------- Paralelismo (passo 2-3 vetorizado) ----------
    workers: int = 0                              # 0 = nº de núcleos - 2
    buildings_per_chunk: int = 8192               # prédios por tarefa de worker
    # ---------- Treino (passo 6) ----------
    loader_workers: int = 16
    use_amp: bool = True                          # mixed precision no treino

    # ---------- Passo 2: janelamento ----------
    # ATENÇÃO (resolução): valores em PASSOS. Para "15min", 7 dias = 672,
    # forecast/stride/period/val_horizon = 96. Ajuste ao trocar resolution.
    backcast_length: int = 168                    # 7 dias (horário) — igual ao W-LSTMix
    forecast_length: int = 24
    stride: int = 24                              # sobreposição: janelas deslizam 1 dia
    period: int = 24

    # ---------- Passo 3: wavelet ----------
    wavelet: str = "db4"
    wavelet_level: int = 5                        # será truncado por dwt_max_level se preciso
    # Sharding do passo 2-3 (proteção de RAM p/ grupos gigantes, ex.
    # Buildings-900K): descarrega um .pt a cada N janelas acumuladas,
    # sempre em fronteira de prédio. ~3,9 KB/janela -> 200k ~ 0,8 GB.
    max_windows_per_shard: int = 200_000

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