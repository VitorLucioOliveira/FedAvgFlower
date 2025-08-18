# fed-avg: A Flower / PyTorch app

## Install dependencies and project

```bash
pip install -e .
```

## Como rodar a simulação principal e gerar gráficos

### 1. Executando a simulação principal (FedAvg)

Para rodar a simulação principal do FedAvg, utilize o comando abaixo. Ele executa a simulação usando o Flower e salva toda a saída (logs) no arquivo `simulation_log.txt`:

```bash
flwr run . > simulation_log.txt 2>&1
```

### 2. Gerando os gráficos a partir do log

Após rodar a simulação, execute o script para gerar os gráficos:

```bash
python plot_results.py
```

O script irá:

- Ler as configurações do `pyproject.toml`
- Gerar e salvar os gráficos na pasta de saída correspondente aos parâmetros da simulação.

### 3. Base de Dados Utilizada

O código utiliza o dataset **Fashion MNIST** (via `zalando-datasets/fashion_mnist`), que contém imagens de roupas classificadas em 10 categorias.

### 4. Gráficos Gerados

Ao rodar o `plot_results.py`, serão gerados os seguintes gráficos:

- **Distribuição de Rótulos por Cliente** (`label_distribution_alpha_X.png`): mostra como as classes do dataset foram distribuídas entre os clientes após a partição (útil para visualizar o grau de heterogeneidade dos dados).
- **Acurácia vs. Rounds** (`accuracy_vs_rounds.png`): mostra a evolução da acurácia média do modelo global ao longo das rodadas de federated learning.
- **Loss vs. Rounds** (`loss_vs_rounds.png`): mostra a evolução da função de perda média ao longo das rodadas.

Os arquivos são salvos na pasta `out-put/`, em um subdiretório que indica os principais parâmetros da simulação.

### 5. Significado do Nome da Pasta de Saída

O nome da pasta de saída segue o padrão:

```
C{NUM_PARTITIONS}_A{ALPHA}_E{EPOCHS}_CF{CLIENT_FIT}
```

Onde:

- **C**: Número de clientes/partições (ex: `C100` = 100 clientes)
- **A**: Valor de alpha usado na partição de Dirichlet (ex: `A0.1` = alpha 0.1, controla o grau de heterogeneidade)
- **E**: Número de épocas locais de treinamento em cada cliente (ex: `E5` = 5 épocas)
- **CF**: Fração de clientes selecionados por rodada (ex: `CF0.1` = 10% dos clientes participam por rodada)

Exemplo:
`out-put/C100_A0.1_E5_CF0.1`
significa: 100 clientes, alpha=0.1, 5 épocas locais, 10% dos clientes participando por rodada.
