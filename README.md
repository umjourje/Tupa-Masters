# Aplicação de Aprendizado Federado para Classificação de Anomalias com Dados Non-IID no Contexto Elétrico

Este projeto estende o **[W-LSTMix](https://github.com/EdgeIntelligenceLab/W-LSTMix)** — um modelo híbrido leve de *forecasting* de carga elétrica (N-BEATS + LSTM + MLP-Mixer com decomposição wavelet) — em duas frentes complementares, sobre o dataset **[EnergyBench](https://huggingface.co/datasets/ai-iot/EnergyBench)**:

1. **Pipeline de detecção de anomalias sem *data leakage*** (centralizado): decomposição wavelet **por janela**, rotulagem bicaudal não supervisionada, modelo híbrido *forecasting* + classificação e validação *rolling-origin*;
2. **Camada de Aprendizado Federado** (Flower ≥ 1.21, Message API): treinamento distribuído do W-LSTMix em clientes físicos (Raspberry Pi), onde cada dispositivo detém apenas a sua partição de dados — o cenário **Non-IID** real do domínio elétrico (edifícios/regiões com distribuições de consumo heterogêneas).

> ⚠️ **Status de integração (leia antes de usar)**
>
> | Camada | Estado | Observação |
> |---|---|---|
> | Pipeline anti-leak (passos 1–8) | ✅ implementado | wavelet por janela, rótulos q1/q99, treino híbrido, rolling-origin |
> | Camada federada (Flower App) | ✅ implementado | **forecasting apenas**, com o pré-processamento *original* do W-LSTMix (decomposição sobre a série inteira de cada split) |
> | Federado × pipeline anti-leak + classificação de anomalias | 🔜 roadmap | requer o `task.py` consumir os artefatos `.pt` janelados/rotulados e o modelo híbrido |
>
> Ou seja: os resultados federados atuais são comparáveis ao W-LSTMix centralizado original; a versão federada da detecção de anomalias sem leak é o próximo marco.

---

## 📊 Dados

Perfis **reais** de carga do EnergyBench (`Energy-Load-Profiles`), resoluções 15min/30min/Hourly, setores `Commercial` e `Residential`, com dezenas de agrupamentos por país/dataset (BDG-2, Enernoc, Prayas, GoiEner, SGSC, CEEW, MFRED, entre outros).

O EnergyBench **não** segue o padrão "1 arquivo = 1 prédio": há arquivos *wide* (uma coluna por medidor), *long* (coluna de id do prédio) e particionados por ano. O passo 1 do pipeline normaliza tudo para o contrato `1 parquet = 1 prédio` com coluna `energy` — o mesmo contrato esperado pelo `task.py` federado.

> ⚠️ Os datasets são usados sob seus respectivos termos/licenças, exclusivamente para pesquisa acadêmica.

---

## 🔁 Pipeline centralizado (anti-leak)

```
pipeline/
├── step1_split.py                 # 1. Extração por prédio + split temporal TREINO/TESTE
├── step2_3_windows_wavelet.py     # 2-3. Sliding window (overlap) + wavelet POR JANELA -> .pt
├── step4_5_labels.py              # 4-5. Rótulos bicaudais (q1/q99 intra-janela) + fusão -> coluna `anomaly`
├── step6_train.py                 # 6-7. Treino híbrido (forecast + classificação) com validação rolling-origin
└── step8_test.py                  # 8. Teste com rótulos gerados EM RUNTIME (mesmos passos 3-5)
```

Propriedades anti-leak do desenho:

1. **Split antes de tudo** — corte cronológico por prédio; não existe split estático de validação;
2. **Wavelet por janela** — `pywt.wavedec` recebe apenas os `backcast+forecast` pontos da janela;
3. **Rótulos intra-janela** — quantis q1/q99 do resíduo calculados dentro de cada janela; fusão (`any`/`majority`) entre janelas sobrepostas;
4. **Validação rolling-origin** — no *fold* j, valida-se o dia `[c_j, c_j+24)`; o dia validado entra no treino do *fold* j+1 (*warm start*);
5. **Teste rotulado em runtime** — rótulos do teste gerados na avaliação pelas MESMAS funções dos passos 3-5.

### Saídas

```
<out_root>/
├── 01_splits/<res>/<split>/<Sector>/<Grupo>/<prédio>.parquet
├── 02_windows/<res>/<split>/<Sector>/<Grupo>.pt
├── 03_labeled_series/<res>/<split>/<Sector>/<Grupo>/...
└── 04_models/
```

---

## 🌸 Camada federada (Flower ≥ 1.21, Message API)

Implementada como um **Flower App** (`ServerApp`/`ClientApp`, execução via `flwr run`), voltada a **clientes físicos** — cada Raspberry Pi contém apenas a sua partição de dados no disco local (não há particionador de simulação: *a partição É o disco*).

```
federated/
├── task.py          # modelo, dados, treino e avaliação locais (adaptação do train.py/test.py)
├── client_app.py    # ClientApp: lê data_root/partition de context.node_config
├── server_app.py    # ServerApp: modelo inicial, estratégia, strategy.start(...)
├── strategy.py      # TensorBoardFedAvg (FedAvg + logging por rodada)
├── models/          # W_LSTMix.py (copiado do repositório original, inalterado)
├── my_utils/        # metrics.py do paper (CVRMSE, NRMSE, MAE, MSE)
└── configs/         # W_LSTMix.json — IDÊNTICO em todos os nós (FedAvg exige arquiteturas iguais)
```

Decisões de projeto da adaptação federada (documentadas no `task.py`):

- **Treino local como função pura** que retorna métricas numéricas num `MetricRecord` (`train_loss`, curva `train_loss_epochs`, `val_loss` opcional, `num-examples`);
- **Sem early stopping/checkpoint local**: a decisão de "melhor modelo" pertence ao **servidor** (estratégia), que compara métricas agregadas por rodada;
- **Ponderação do FedAvg** pela chave `num-examples` (nº de janelas locais) — o padrão `weighted_by_key` da estratégia;
- **Métricas do paper** importadas de `my_utils/metrics.py` para comparabilidade com os resultados centralizados (com *fallback* sinalizado caso o import falhe);
- **Avaliação local do modelo global** na partição de teste de cada cliente, agregada por partição;
- **`TensorBoardFedAvg`**: subclasse de `flwr.serverapp.strategy.FedAvg` que sobrescreve `aggregate_train`/`aggregate_evaluate` para registrar as métricas agregadas de cada rodada em TensorBoard (`tb_logs/server`).

### Estrutura de dados esperada em cada cliente

```
<data_root>/
├── train/ <região>/<prédio>.csv|.parquet    # coluna 'energy'
├── val/   <região>/<prédio>.csv|.parquet    # opcional
└── test/  <região>/<prédio>.csv|.parquet
```

O passo 1 do pipeline centralizado gera exatamente este formato — basta distribuir a cada dispositivo o subconjunto de grupos que constitui a sua partição (ex.: 1 cliente = 1 país/dataset).

### Execução

```bash
# em cada dispositivo/nó e no servidor, dentro do diretório do Flower App:
flwr run .

# acompanhamento das métricas agregadas por rodada:
tensorboard --logdir tb_logs/server
```

---

## 📈 Visualização e inspeção (pipeline centralizado)

Scripts com seleção interativa de **grupo (país) → prédio**, sempre lendo os artefatos que **realmente rodaram**:

| Script | Inspeciona | Conteúdo |
|---|---|---|
| `plot1_split.py` | passo 1 | série completa com assíntota vertical na fronteira real treino/teste |
| `plot2_decomposition.py` | passos 2-3 e 4 | (A) decomposição trend/sazonal por janela; (B) banda de anomalia `[trend+q01, trend+q99]` com pontos fora da banda |
| `plot3_windowing.py` | passos 4-5 | frames do janelamento (janela, stride, validação de 1 dia) sobre a série rotulada, com anomalias fundidas |

Todos aceitam `--list`, `--group`/`--building` e `--split`.

---

## 🛠 Instalação

```bash
git clone <este-repositório>
cd <este-repositório>
pip install torch pywavelets pandas pyarrow numpy scikit-learn seaborn matplotlib tqdm statsmodels
pip install "flwr>=1.21" tensorboard          # camada federada

# backbone original (copiar models/, my_utils/ e configs/ para federated/):
git clone https://github.com/EdgeIntelligenceLab/W-LSTMix

# dados:
git clone https://huggingface.co/datasets/ai-iot/EnergyBench
```

Configuração: `pipeline/config.py` (caminhos, janela/stride, quantis, rolling-origin) e `federated/configs/W_LSTMix.json` (hiperparâmetros do modelo — **idêntico em todos os nós**).

---

## 🗺 Roadmap

- [x] Extração/normalização do EnergyBench (wide/long/single, partições por ano)
- [x] Janelamento com sobreposição + decomposição wavelet por janela (anti-leak)
- [x] Rotulagem bicaudal intra-janela + fusão entre janelas
- [x] Modelo híbrido (forecast + classificação) com validação rolling-origin
- [x] Avaliação com rótulos gerados em runtime
- [x] Flower App (Message API ≥ 1.21) para treino federado do W-LSTMix em clientes físicos
- [x] Estratégia FedAvg com logging TensorBoard por rodada
- [ ] `task.py` consumindo os artefatos anti-leak (`02_windows/*.pt` + `labels_fused`) no lugar da decomposição sobre a série inteira
- [ ] Cabeça de classificação de anomalias no cliente federado (modelo híbrido federado)
- [ ] Estratégias robustas a Non-IID (FedProx, SCAFFOLD) e análise do impacto da heterogeneidade
- [ ] Comparação federado × centralizado × local-only

---

## 🙏 Créditos e citação

Este projeto se apoia no W-LSTMix, no EnergyBench e no framework [Flower](https://flower.ai). Se usar este código, cite também o trabalho original:

```bibtex
@inproceedings{dwivedi2025wlstmix,
  title={W-{LSTM}ix: A Hybrid Modular Forecasting Framework for Trend and Pattern Learning in Short-Term Load Forecasting},
  author={Shivam Dwivedi and Anuj Kumar and Harish Kumar Saravanan and Pandarasamy Arjunan},
  booktitle={1st ICML Workshop on Foundation Models for Structured Data},
  year={2025},
  url={https://openreview.net/forum?id=bG04Z3Jioc}
}
```

Implementação de referência anterior (com decomposição pré-split, motivadora da correção anti-leak): [W-LSTMix-Anomaly-Detection](https://github.com/stepsbtw/W-LSTMix-Anomaly-Detection).
