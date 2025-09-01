# Fed-Avg: Aplicação de Federated Learning com Flower e PyTorch

Este projeto implementa um sistema de **Federated Learning** usando o framework **Flower** e **PyTorch**, aplicado ao dataset **Fashion MNIST**. O sistema permite executar simulações de aprendizado federado com diferentes configurações de hiperparâmetros e gerar visualizações dos resultados.

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.8+
- pip

### Instalação das dependências

```bash
pip install -e .
```

## 📁 Estrutura do Projeto

```
fed-avg/
├── fed_avg/                 # Código principal da aplicação
│   ├── __init__.py
│   ├── client_app.py        # Implementação do cliente federado
│   ├── server_app.py        # Implementação do servidor federado
│   └── task.py              # Configurações da tarefa de ML
├── out-put/                 # Pasta de saída dos resultados
├── plot_results.py          # Script para geração de gráficos
├── run_experiments.py       # Script para execução de múltiplos experimentos
├── pyproject.toml           # Configurações do projeto
└── README.md
```

## 🔬 Executando Experimentos

### 1. Simulação Única (FedAvg)

Para executar uma simulação única do FedAvg:

```bash
flwr run . > simulation_log.txt 2>&1
```

### 2. Múltiplos Experimentos com Otimização Aleatória

O arquivo `run_experiments.py` permite executar múltiplos experimentos com diferentes combinações de hiperparâmetros de forma automatizada:

#### Funcionalidades:

- **Otimização Aleatória**: Seleciona aleatoriamente valores de hiperparâmetros de listas predefinidas
- **Execução Automatizada**: Roda múltiplas simulações consecutivas
- **Logging Automático**: Salva automaticamente os logs de cada simulação
- **Geração de Gráficos**: Chama automaticamente o script de plotagem após cada simulação

#### Hiperparâmetros Configuráveis:

- **`fraction_fit`**: Fração de clientes que participam por rodada (0.1 a 1.0)
- **`local_epochs`**: Número de épocas locais de treinamento (1 a 20)
- **`learning_rate`**: Taxa de aprendizado (0.001 a 0.01)
- **`batch_size`**: Tamanho do batch (10 a 50)

#### Uso:

```bash
python run_experiments.py
```

Por padrão, executa experimentos com combinações aleatórias de hiperparâmetros.

### 3. Geração de Gráficos e Análise

O arquivo `plot_results.py` é responsável por processar os logs das simulações e gerar visualizações completas:

#### Funcionalidades Principais:

##### **Análise de Distribuição de Dados:**

- Gera gráficos mostrando como os rótulos do Fashion MNIST são distribuídos entre os clientes
- Utiliza partição Dirichlet com alpha configurável (padrão: 100)
- Visualiza o grau de heterogeneidade dos dados entre clientes

##### **Processamento de Métricas:**

- Extrai automaticamente métricas de loss e acurácia dos logs de simulação
- Limpa códigos ANSI e formata os dados para análise
- Calcula estatísticas como melhor loss/acurácia e rounds correspondentes

##### **Geração de Gráficos:**

- **Acurácia vs. Rounds**: Evolução da acurácia média ao longo das rodadas
- **Loss vs. Rounds**: Evolução da função de perda ao longo das rodadas
- **Distribuição de Rótulos**: Visualização da heterogeneidade dos dados

##### **Organização de Resultados:**

- Cria pastas organizadas por parâmetros da simulação
- Salva resultados em CSV para análise posterior
- Nomenclatura automática: `CF{fraction_fit}_E{epochs}_BS{batch_size}_LR{learning_rate}`

`

## 📈 Interpretação dos Resultados

### Estrutura das Pastas de Saída

```
out-put/
├── CF0.4_E15_BS33_LR0.01/     # Exemplo de pasta de resultados
│   ├── accuracy_vs_rounds.png
│   └── loss_vs_rounds.png
└── results_summary.csv          # Resumo consolidado de todos os experimentos
```

### Significado dos Parâmetros:

- **CF**: Client Fraction (fração de clientes por rodada)
- **E**: Local Epochs (épocas locais de treinamento)
- **BS**: Batch Size (tamanho do batch)
- **LR**: Learning Rate (taxa de aprendizado)

## 🔧 Personalização

### Modificando Parâmetros dos Experimentos

Edite `run_experiments.py` para alterar:

- Número de experimentos (`num_testes`)
- Ranges dos hiperparâmetros
- Estratégia de seleção de parâmetros

### Configurações do Dataset

Modifique `plot_results.py` para ajustar:

- Valor de alpha para partição Dirichlet
- Número de partições
- Dataset utilizado

## 📝 Logs e Debugging

- **Logs de Simulação**: Salvos em `simulation_log.txt`
- **Logs de Plotagem**: Exibidos no console durante execução
- **Tratamento de Erros**: Inclui verificações robustas para arquivos e dados

## 🤝 Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
