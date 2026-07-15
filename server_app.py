"""server_app.py — ServerApp Flower (Message API) para o W-LSTMix federado.

Substitui o antigo server.py (fl.server.start_server + SaveModelStrategy).
Roda dentro do SuperLink na máquina servidora:

    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key

    # e, para submeter a execução:
    flwr run . raspberry-deployment

O registro de métricas por rodada (TensorBoard) e o checkpoint do melhor
modelo global ficam a cargo da TensorBoardFedAvg (strategy.py). Visualize:

    tensorboard --logdir tb_logs/server
"""

from __future__ import annotations

import logging

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from meuapp import task                       # ajuste 'meuapp' ao seu pacote
from meuapp.strategy import TensorBoardFedAvg

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
    arrays = ArrayRecord(model.state_dict())

    strategy = TensorBoardFedAvg(
        fraction_train=1.0,        # com poucos Pis, use todos a cada rodada
        fraction_evaluate=1.0,
        log_dir="tb_logs/server",
        checkpoint_path="best_model_global.pth",
        selection_metric="test_loss",   # ou "nrmse"/"cvrmse", conforme o estudo
        lower_is_better=True,
        local_epochs=local_epochs,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Modelo da ÚLTIMA rodada (o MELHOR já foi salvo pela estratégia)
    torch.save(result.arrays.to_torch_state_dict(), "final_model_global.pth")
    log.info(
        "Execução concluída: final_model_global.pth (última rodada) e "
        "best_model_global.pth (melhor rodada) salvos."
    )
