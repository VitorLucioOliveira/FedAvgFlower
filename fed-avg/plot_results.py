import re
import matplotlib.pyplot as plt
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
import os
import tomllib
import csv

with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)

# --- Configurações ---

DATASET = "zalando-datasets/fashion_mnist"
ALPHA = 100
NUM_PARTITIONS = 100


def clean_ansi_codes(text):
    """Remove códigos de escape ANSI (cores) de uma string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def plot_partition_label_distribution(output_dir):
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
        full_path = os.path.join(output_dir, distribution_filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Gráfico de distribuição de rótulos salvo em: '{full_path}'")
        plt.close(fig)

    except Exception as e:
        print(f"Erro ao gerar o gráfico de distribuição de rótulos: {e}")


def plot_metrics_from_log(log_file_path, fraction_fit, local_epochs, batch_size, learning_rate ):
    """
    Lê o log, extrai métricas e salva os gráficos na pasta de saída.
    """
    output_dir = (f"out-put/CF{fraction_fit}"
                   f"_E{local_epochs}"
                   f"_BS{batch_size}"
                   f"_LR{learning_rate}")

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo '{log_file_path}' não foi encontrado.")
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

        # Extrai a seção de Acurácia
        accuracy_section_match = re.search(r"History \(metrics, distributed, evaluate\):([\s\S]*)", log_content)
        if not accuracy_section_match:
            raise AttributeError("Seção de Acurácia não encontrada.")
        accuracy_section = accuracy_section_match.group(1)
        accuracy_matches = re.findall(r"\((\d+), ([\d.]+)\)", accuracy_section)

    except AttributeError as e:
        print(f"Erro: Não foi possível encontrar as seções de métricas no log. Detalhe: {e}")
        return

    if not loss_matches or not accuracy_matches:
        print("Nenhuma métrica de loss ou acurácia foi encontrada após a análise.")
        return

    rounds = [int(r) for r, _ in loss_matches]
    losses = [float(l) for _, l in loss_matches]
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

    # --- Estatísticas ---
    best_loss = metrics_df["Loss"].min()
    best_loss_round = metrics_df.loc[metrics_df["Loss"].idxmin(), "Round"]
    final_loss = metrics_df["Loss"].iloc[-1]

    best_acc = metrics_df["Accuracy"].max()
    best_acc_round = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Round"]
    final_acc = metrics_df["Accuracy"].iloc[-1]

    # --- Gráfico de Acurácia ---
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Round"], metrics_df["Accuracy"], marker="o", linestyle="-", color="b")
    plt.title(f"Acurácia Média por Round")
    plt.xlabel("Round")
    plt.ylabel("Acurácia")
    plt.grid(True)

    # Textos
    plt.text(0.92, 1.1, f"Best: {best_acc:.2f} (Round {best_acc_round})",
             transform=plt.gca().transAxes, fontsize=10, color="green", ha="left")
    plt.text(0.92, 1.05, f"Final: {final_acc:.2f}",
             transform=plt.gca().transAxes, fontsize=10, color="blue", ha="left")

    acc_path = os.path.join(output_dir, "accuracy_vs_rounds.png")
    plt.savefig(acc_path)
    plt.close()

    # --- Gráfico de Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["Round"], metrics_df["Loss"], marker="o", linestyle="-", color="r")
    plt.title(f"Loss Médio por Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)

    # Textos
    plt.text(0.92, 1.1, f"Best: {best_loss:.2f} (Round {best_loss_round})",
             transform=plt.gca().transAxes, fontsize=10, color="green", ha="left")
    plt.text(0.92, 1.05, f"Final: {final_loss:.2f}",
             transform=plt.gca().transAxes, fontsize=10, color="blue", ha="left")

    loss_path = os.path.join(output_dir, "loss_vs_rounds.png")
    plt.savefig(loss_path)
    plt.close()

    # --- Salvar resultados finais em CSV ---
    results_csv = os.path.join("out-put", "results_summary.csv")

    # Verifica se já existe
    file_exists = os.path.isfile(results_csv)

    with open(results_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Se o arquivo acabou de ser criado, escreve o cabeçalho
        if not file_exists:
            writer.writerow([
                "clients", "rounds", "fraction_fit", "local_epochs",
                "batch_size", "learning_rate",
                "best_loss", "final_loss",
                "best_acc", "final_acc"
            ])

        # Adiciona a linha de resultados
        writer.writerow([
            NUM_PARTITIONS, len(rounds), fraction_fit, local_epochs,
            batch_size, learning_rate,
            round(best_loss, 4), round(final_loss, 4),
            round(best_acc, 4), round(final_acc, 4)
        ])

    print(f"Linha adicionada à tabela em: '{results_csv}'")

    print(f"Gráficos de acurácia e loss salvos em: '{output_dir}'")



