"""server_app.py — ServerApp Flower (Message API) para o W-LSTMix federado.

FUSÃO: mantém a SUA versão (fractions=1.0, TensorBoardFedAvg com checkpoint
do melhor global por selection_metric, TLS via SuperLink, salvamento de
final + best, pacote do app) + o bloco do "v0" decidido neste chat: o
modelo da rodada 1 NÃO é aleatório — são os pesos do treino CENTRALIZADO
no sintético (passo 6, máquina grande), carregados com strict=True.

Execução (máquina servidora):

    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key

    flwr run . raspberry-deployment

TensorBoard: tensorboard --logdir tb_logs/server
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

import task                                   # mesmo diretório do app
from strategy import TensorBoardFedAvg

log = logging.getLogger("wlstmix.server")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    local_epochs = int(context.run_config.get("local-epochs", 1))

    # Modelo global inicial — MESMA config usada pelos clientes
    cfg = task.load_config()
    device = torch.device("cpu")
    model = task.get_model(cfg, device)

    # LINHA CRUCIAL (v0): pesos do treino centralizado no sintético
    # instanciados na rodada 1. strict=True: divergência de arquitetura
    # entre v0 e os clientes falha AQUI, não na rodada 3.
    v0 = str(context.run_config.get("v0-path", ""))
    if v0 and Path(v0).exists():
        model.load_state_dict(torch.load(v0, map_location=device),
                              strict=True)
        log.info("v0 carregado de %s", v0)
    else:
        log.warning("v0-path %r não encontrado — iniciando de pesos "
                    "ALEATÓRIOS (ok só para ensaio).", v0)

    arrays = ArrayRecord(model.state_dict())

    strategy = TensorBoardFedAvg(
        fraction_train=1.0,        # com poucos Pis, use todos a cada rodada
        fraction_evaluate=1.0,
        log_dir="tb_logs/server",
        checkpoint_path="best_model_global.pth",
        selection_metric="test_loss",   # ou "nrmse"/"cvrmse"/"f1"
        lower_is_better=True,
        local_epochs=local_epochs,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Modelo da ÚLTIMA rodada (o MELHOR já foi salvo pela estratégia)
    tmp = Path("final_model_global.pth.tmp")
    torch.save(result.arrays.to_torch_state_dict(), tmp)
    tmp.replace("final_model_global.pth")
    log.info("Execução concluída: final_model_global.pth (última rodada) e "
             "best_model_global.pth (melhor rodada) salvos.")