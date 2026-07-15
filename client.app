"""client_app.py — ClientApp Flower (Message API) para o W-LSTMix federado.

Substitui o antigo client.py (NumPyClient + fl.client.start_client).
Roda dentro de um SuperNode em cada Raspberry Pi:

    flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink IP_DO_SERVIDOR:9092 \
        --node-config "partition-id=0 num-partitions=5 data-root='/home/pi/data'"

Toda a lógica de modelo/dados/treino/avaliação vive em task.py.
"""

from __future__ import annotations

import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from meuapp import task  # ajuste 'meuapp' ao nome do seu pacote

log = logging.getLogger("wlstmix.client")

app = ClientApp()


def _setup(context: Context):
    """Configuração comum a train e evaluate."""
    cfg = task.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition_id = int(context.node_config["partition-id"])
    data_root = str(context.node_config.get("data-root", "./data"))
    return cfg, device, partition_id, data_root


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, device, partition_id, data_root = _setup(context)
    local_epochs = int(context.run_config.get("local-epochs", 1))

    # Reconstrói o modelo e carrega os pesos GLOBAIS recebidos do servidor
    model = task.get_model(cfg, device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Dados locais deste dispositivo (a partição É o disco local)
    train_ds = task.load_split(data_root, "train", cfg)
    val_ds = task.load_split(data_root, "val", cfg)  # pode ser None
    if train_ds is None:
        raise FileNotFoundError(f"Sem dados de treino em {data_root}/train")

    log.info("Cliente %d: iniciando %d época(s) local(is)", partition_id, local_epochs)
    metrics = task.train(model, train_ds, cfg, device, local_epochs, val_ds)
    metrics["partition_id"] = partition_id  # rotula as métricas no servidor

    reply = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, device, partition_id, data_root = _setup(context)

    model = task.get_model(cfg, device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    metrics = task.evaluate(model, cfg, device, data_root)
    metrics["partition_id"] = partition_id

    reply = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)
