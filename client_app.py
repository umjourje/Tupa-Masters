"""client_app.py — ClientApp (Message API): treino/avaliação locais do
HybridWLSTMix sobre os artefatos do pipeline no disco do dispositivo.

Cada SuperNode declara sua partição:
    flower-supernode ... --node-config "data-root='/dados/particao_X'"

local-epochs e lr vêm do run_config do app (pyproject/flwr run), o mesmo
valor que o servidor usa para o eixo de épocas do TensorBoard.
"""
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import task

app = ClientApp()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = _device()
    model = task.get_model(task.load_config(), device)
    # pesos globais da rodada -> modelo local
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    data_root = Path(context.node_config["data-root"])
    metrics = task.train(
        model, data_root, device,
        epochs=int(context.run_config.get("local-epochs", 1)),
        lr=float(context.run_config.get("lr", 1e-3)))
    reply = RecordDict({"arrays": ArrayRecord(model.state_dict()),
                        "metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = _device()
    model = task.get_model(task.load_config(), device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.eval()
    data_root = Path(context.node_config["data-root"])
    metrics = task.evaluate(model, data_root, device)
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}),
                   reply_to=msg)