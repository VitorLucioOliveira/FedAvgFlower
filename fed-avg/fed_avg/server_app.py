"""fed-avg: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import Tuple, List

from torch.utils.data import DataLoader
import json

from fed_avg.task import Net, get_weights, set_weights, test, get_transform
from datasets import load_dataset


def get_avaluate_fn(testloader, device):
    """Returna um callback que avalia o modelo global"""
    def evaluate(server_round, parameters_ndarrays, config):
        """Avalia o modelo global usando um set de teste centralizado"""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate

def weighted_avarage(metrics: List[Tuple[int, Metrics]])-> Metrics:
    """Calcula a metrica de accuracy media do round"""

    accuracies = [num_examples * m ["accuracy"] for num_examples, m in metrics]# multiplicando a accuracy por cada exemplo de treino fo client
    total_examples = sum(num_examples for num_examples, m in metrics)# calculado o número total de exemplos

    return {"accuracy": sum(accuracies) / total_examples}


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]])-> Metrics:
    """Handle metrics from fit method in clients"""
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metrics =  json.loads(my_metric_str)
        b_values.append(my_metrics["b"])

    return {"max_b": max(b_values)}




def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    def on_fit_config(server_round: int) -> Metrics:
        """Alterar o learning rate, baseado no round"""
        lr = context.run_config["learning-rate"]
        # if server_round > 30:
        #     lr = 0.005
        return {"lr": lr}

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    testloader = DataLoader(testset.with_transform(get_transform()))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=10,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_avarage,#returna a accuracy media dos modelos
        # fit_metrics_aggregation_fn=handle_fit_metrics,#retorna metricas extras por round
        on_fit_config_fn=on_fit_config,#mudar configurações durante os rounds
        # evaluate_fn= get_avaluate_fn(testloader, device="cpu"),#avaliar modelo global
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
