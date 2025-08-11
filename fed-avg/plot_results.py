import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
import sys
import subprocess
import os
import tomllib

with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)

# --- Configurações ---
NUM_PARTITIONS = 100
LOG_FILE_PATH = "simulation_log.txt"
OUTPUT_DIR = "out-put"
DATASET = "cifar10"
ALPHA = 100
EPOCHS = config["tool"]["flwr"]["app"]["config"]["local-epochs"]
CLIENT_FIT = config["tool"]["flwr"]["app"]["config"]["fraction-fit"]
ROUNDS = config["tool"]["flwr"]["app"]["config"]["num-server-rounds"]



def clean_ansi_codes(text):
    """Remove códigos de escape ANSI (cores) de uma string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def plot_partition_label_distribution():
    """
    Cria e salva um gráfico mostrando a distribuição de rótulos de dados
    em cada partição (cliente), garantindo que os nomes dos rótulos apareçam.
    """
    print(f"Gerando gráfico de distribuição de rótulos (com alpha={ALPHA})...")

    try:
        fds = FederatedDataset(
            dataset=DATASET,
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=NUM_PARTITIONS,
                    partition_by="label",
                    alpha=ALPHA,
                    seed=42,
                    min_partition_size=0,
                ),
            },
        )
        partitioner = fds.partitioners["train"]
        label_names = fds.load_split("train").features["label"].names

    except Exception as e:
        print(f"Erro ao inicializar o particionador do dataset: {e}")
        return

    try:
        fig, ax, df = plot_label_distributions(
            partitioner=partitioner,
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=label_names,
            title=f"Distribuição de Rótulos por Cliente (alpha={ALPHA})",
        )
        fig.set_size_inches(14, 8)
        fig.tight_layout(rect=[0, 0, 0.99, 1])

        distribution_filename = f"label_distribution_alpha_{ALPHA}.png"
        full_path = os.path.join(OUTPUT_DIR, distribution_filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Gráfico de distribuição de rótulos salvo em: '{full_path}'")
        plt.close(fig)

    except Exception as e:
        print(f"Erro ao gerar o gráfico de distribuição de rótulos: {e}")


def plot_metrics_from_log():
    """
    Lê o log, extrai métricas e salva os gráficos na pasta de saída.
    """
    print("\nGerando gráficos de métricas a partir do log...")

    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{LOG_FILE_PATH}' não foi encontrado.")
        return

    log_content = clean_ansi_codes(raw_content)

    try:
        # Extrai a seção de Loss
        loss_section_match = re.search(
            r"History \(loss, distributed\):([\s\S]*?)History \(metrics, distributed, evaluate\):", log_content)
        if not loss_section_match:
            raise AttributeError("Seção de Loss não encontrada.")
        loss_section = loss_section_match.group(1)
        loss_matches = re.findall(r"round (\d+): ([\d.]+)", loss_section)

        # --- CORREÇÃO APLICADA AQUI ---
        # Extrai a seção de Acurácia
        accuracy_section_match = re.search(r"History \(metrics, distributed, evaluate\):([\s\S]*)", log_content)
        if not accuracy_section_match:
            raise AttributeError("Seção de Acurácia não encontrada.")
        accuracy_section = accuracy_section_match.group(1)
        # Encontra todos os pares (round, acurácia) dentro da seção, ignorando o resto
        accuracy_matches = re.findall(r"\((\d+), ([\d.]+)\)", accuracy_section)

    except AttributeError as e:
        print(f"Erro: Não foi possível encontrar as seções de métricas no log. Detalhe: {e}")
        return

    if not loss_matches or not accuracy_matches:
        print("Nenhuma métrica de loss ou acurácia foi encontrada após a análise.")
        return

    rounds = [int(r) for r, _ in loss_matches]
    losses = [float(l) for _, l in loss_matches]
    # A acurácia já vem em pares (round, valor), então pegamos só o valor
    accuracies = [float(a) for _, a in accuracy_matches]

    min_len = min(len(rounds), len(losses), len(accuracies))
    if min_len == 0:
        print("Não foram extraídos dados de métricas suficientes.")
        return

    metrics_df = pd.DataFrame({
        "Round": rounds[:min_len],
        "Loss": losses[:min_len],
        "Accuracy": accuracies[:min_len]
    })

    # Gráfico de Acurácia
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Round"], metrics_df["Accuracy"], marker="o", linestyle="-", color="b")
    plt.title(f"Acurácia Média por Round")
    plt.xlabel("Round")
    plt.ylabel("Acurácia")
    plt.grid(True)
    acc_path = os.path.join(OUTPUT_DIR, f"accuracy_vs_rounds({ROUNDS}-{CLIENT_FIT}-{EPOCHS}).png")
    plt.savefig(acc_path)
    plt.close()

    # Gráfico de Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Round"], metrics_df["Loss"], marker="o", linestyle="-", color="r")
    plt.title(f"Loss Médio por Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_path = os.path.join(OUTPUT_DIR, f"loss_vs_rounds({ROUNDS}-{CLIENT_FIT}-{EPOCHS}).png")
    plt.savefig(loss_path)
    plt.close()

    print(f"Gráficos de acurácia e loss salvos em: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import seaborn
    except ImportError:
        print("Instalando bibliotecas de plotagem (seaborn, matplotlib)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "matplotlib"])

    plot_partition_label_distribution()
    plot_metrics_from_log()