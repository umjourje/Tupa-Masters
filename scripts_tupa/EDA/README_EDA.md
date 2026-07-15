# EDA — EnergyBench (ai-iot/EnergyBench)

Dois scripts independentes, um por pasta do dataset (`Dataset_V0.0/`):

| Script | Pasta-alvo | Estratégia |
|---|---|---|
| `eda_real_energy_load_profiles.py` | `Energy-Load-Profiles/` (~3,35 GB) | Em memória, por sub-dataset — estilo do repo `pierre-haessig/ausgrid-solar-data` |
| `eda_synthetic_energy_load_profiles.py` | `Synthetic-Energy-Load-Profiles/` (centenas de GB) | Out-of-core (pyarrow, streaming) + amostra visual comparável ao real |

## Pré-requisitos

```bash
pip install pandas numpy matplotlib pyarrow
# download do dataset (exemplos):
# huggingface-cli download ai-iot/EnergyBench --repo-type dataset \
#     --include "Dataset_V0.0/Energy-Load-Profiles/Hourly/*" --local-dir .
```

## Uso — dados reais

```bash
# todos os sub-datasets horários
python eda_real_energy_load_profiles.py --root ./Dataset_V0.0 --resolution Hourly

# apenas um sub-dataset (ex.: Enernoc), limitando arquivos p/ teste rápido
python eda_real_energy_load_profiles.py --root ./Dataset_V0.0 \
    --resolution Hourly --dataset Enernoc --max-files 2
```

Saídas por sub-dataset (em `eda_outputs_real/<Dataset>/`): série de um edifício
(3 dias), N edifícios aleatórios, distribuições (pico, razão energia/pico em
h/ano — métrica do repo Ausgrid, completude), matriz de disponibilidade,
perfil diário (útil × fim de semana), heatmap mês×hora, energia diária,
correlação entre edifícios; e `SUMMARY_real.csv` consolidado.

## Uso — dados sintéticos

```bash
# passada de reconhecimento: inventário completo + estatísticas de amostra
python eda_synthetic_energy_load_profiles.py --root ./Dataset_V0.0 \
    --resolution Hourly --max-batches 2000

# varredura completa (custosa!) de uma subpasta/estado
python eda_synthetic_energy_load_profiles.py --root ./Dataset_V0.0 \
    --resolution Hourly --subset <nome_da_subpasta> --max-batches 0
```

Camadas: **A** inventário só por metadados Parquet (linhas, tamanhos,
esquemas — custo ~zero); **B** estatísticas em streaming (média/desvio via
Welford por chunks, quantis por reservoir sampling, perfis dia-da-semana×hora
e mensal acumulados incrementalmente); **C** amostra de arquivos/séries para
gráficos comparáveis aos do script real (validação sintético × real antes do
pré-treino do baseline).

## Observações

- Os ~60 sub-datasets não têm esquema único; a leitura é adaptativa
  (detecta coluna temporal, formato longo × largo e colunas numéricas).
  Se algum sub-dataset falhar, o script registra o aviso e prossegue —
  ajuste as tuplas `TIME/ID/VALUE_CANDIDATES` conforme o esquema real.
- Unidades variam entre sub-datasets (kW × kWh, conforme o card do
  EnergyBench); as estatísticas são internamente consistentes por
  sub-dataset, mas não compare valores absolutos entre datasets sem
  harmonizar unidades.
