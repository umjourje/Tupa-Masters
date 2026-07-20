"""strategy.py — Estratégia customizada: FedAvg + TensorBoard + checkpoint do melhor modelo.

Substitui, no mundo novo do Flower (Message API), a antiga SaveModelStrategy
do repositório Tupã, acrescentando:

  1. Registro por rodada no TensorBoard do SERVIDOR:
       - métricas agregadas de treino e avaliação (train/*, evaluate/*);
       - métricas POR CLIENTE (clients/<id>/*), sem agregação;
       - a curva de épocas locais (train_loss_epochs) de cada cliente,
         projetada num eixo de passos global — resolvendo o ponto
         sinalizado anteriormente: a lista NÃO entra na média do FedAvg,
         é consumida aqui e removida antes da agregação.
  2. Checkpoint do MELHOR modelo global (menor métrica de avaliação
     agregada, por padrão test_loss) — o análogo federado correto do
     early stopping/best_model.pth do train.py original do W-LSTMix.

ATENÇÃO (assinaturas): os métodos aggregate_train/aggregate_evaluate e os
atributos de Message/metadata seguem o padrão documentado nos tutoriais
oficiais da série "Customize a Flower Strategy" (Flower >= 1.21), mas as
assinaturas exatas podem variar entre versões menores. Confirme contra o
template gerado por `flwr new` na SUA versão instalada antes de rodar.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger("wlstmix.strategy")

# Chaves que NÃO devem ser agregadas pelo FedAvg (listas/curvas locais)
NON_AGGREGATABLE_KEYS = ("train_loss_epochs",)


def _client_id(msg: Message) -> str:
    """Identifica o cliente remetente para fins de rotulagem no TensorBoard.

    Nota: o nome exato do campo de origem nos metadados da Message deve ser
    confirmado na sua versão (ex.: msg.metadata.src_node_id). O fallback
    abaixo evita quebra caso o atributo mude de nome.
    """
    meta = getattr(msg, "metadata", None)
    for attr in ("src_node_id", "node_id", "source_node_id"):
        value = getattr(meta, attr, None)
        if value is not None:
            return str(value)
    return "desconhecido"


class TensorBoardFedAvg(FedAvg):
    """FedAvg com registro em TensorBoard e checkpoint do melhor modelo global."""

    def __init__(
        self,
        *args,
        log_dir: str = "tb_logs/server",
        checkpoint_path: str = "best_model_global.pth",
        selection_metric: str = "test_loss",   # métrica agregada de avaliação
        lower_is_better: bool = True,
        local_epochs: int = 1,                 # p/ eixo global da curva de épocas
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_path = checkpoint_path
        self.selection_metric = selection_metric
        self.lower_is_better = lower_is_better
        self.local_epochs = local_epochs
        self._best: Optional[float] = None
        self._latest_arrays: Optional[ArrayRecord] = None  # p/ salvar no melhor round

    # ------------------------------------------------------------------
    # TREINO: log por cliente + remoção de chaves não agregáveis + agregação
    # ------------------------------------------------------------------
    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)

        for msg in replies:
            if not msg.has_content():
                continue  # respostas com erro são tratadas pelo FedAvg
            metrics: MetricRecord = msg.content["metrics"]
            cid = _client_id(msg)

            # (i) Curva de épocas locais -> eixo de passos global, por cliente
            epochs_curve = metrics.get("train_loss_epochs")
            if epochs_curve is not None:
                for ep_idx, loss_val in enumerate(list(epochs_curve)):
                    global_step = (server_round - 1) * self.local_epochs + ep_idx
                    self.writer.add_scalar(
                        f"clients/{cid}/train_loss_epoch", float(loss_val), global_step
                    )
                # (ii) Remove ANTES da agregação: listas não devem entrar na média
                for key in NON_AGGREGATABLE_KEYS:
                    metrics.pop(key, None)

            # (iii) Escalares por cliente (sem agregação), indexados pela rodada
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"clients/{cid}/{key}", float(value), server_round
                    )

        # (iv) Agregação padrão do FedAvg (ponderada por num-examples)
        arrays, agg_metrics = super().aggregate_train(server_round, replies)

        # Guarda referência aos pesos agregados desta rodada para o checkpoint
        self._latest_arrays = arrays

        if agg_metrics is not None:
            for key, value in agg_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", float(value), server_round)
        self.writer.flush()
        return arrays, agg_metrics

    # ------------------------------------------------------------------
    # AVALIAÇÃO: log agregado + por cliente + checkpoint do melhor modelo
    # ------------------------------------------------------------------
    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)

        for msg in replies:
            if not msg.has_content():
                continue
            cid = _client_id(msg)
            for key, value in msg.content["metrics"].items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(
                        f"clients/{cid}/eval_{key}", float(value), server_round
                    )

        agg_metrics = super().aggregate_evaluate(server_round, replies)

        if agg_metrics is not None:
            for key, value in agg_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"evaluate/{key}", float(value), server_round)

            # Checkpoint do melhor modelo global (equivalente federado do
            # best_model.pth do W-LSTMix, decidido pelo SERVIDOR)
            current = agg_metrics.get(self.selection_metric)
            if current is not None and self._latest_arrays is not None:
                current = float(current)
                improved = (
                    self._best is None
                    or (self.lower_is_better and current < self._best)
                    or (not self.lower_is_better and current > self._best)
                )
                if improved:
                    self._best = current
                    # Escrita ATÔMICA (padrão do pipeline): .tmp + rename —
                    # um kill no meio nunca corrompe o melhor checkpoint.
                    tmp = str(self.checkpoint_path) + ".tmp"
                    torch.save(
                        self._latest_arrays.to_torch_state_dict(), tmp)
                    import os as _os
                    _os.replace(tmp, self.checkpoint_path)
                    log.info(
                        "Rodada %d: novo melhor %s=%.6f — checkpoint salvo em %s",
                        server_round, self.selection_metric, current,
                        self.checkpoint_path,
                    )
                    self.writer.add_scalar(
                        f"evaluate/best_{self.selection_metric}", current, server_round
                    )

        self.writer.flush()
        return agg_metrics