import random
import subprocess

import numpy as np

from plot_results import plot_metrics_from_log


num_supernodes = 100


def random_values(limite_inferior, limite_superior):
    """"Com os limites explicitados, gera listas de ordem aleatorias"""

    lista = list(range(limite_inferior, limite_superior+1))
    random.shuffle(lista)

    return lista


def random_optimization(fraction_fit_list, local_epochs_list, learning_rate_list, batch_size_list, num_rounds: int):
    """Pega valores aleatorios dos parametros, faz o treinamento e os graficos dos resultados"""

    for rodada in range(num_rounds):

        fraction_fit =  random.choice(fraction_fit_list) #Fraction fit é em porcentagem
        local_epochs = random.choice(local_epochs_list)
        learning_rate = random.choice(learning_rate_list)
        batch_size = random.choice(batch_size_list)

        # É necessario" " no final de cada hyperparâmetro. Alguns tem que arredondar por algum motivo （；¬＿¬)
        comando = ["flwr", "run", ".",
                   f"--run-config=fraction-fit={round(fraction_fit, 1)} "
                   f"local-epochs={local_epochs} "
                   f"learning-rate={round(learning_rate, 3)} "
                   f"batch-size={batch_size} "]

        print(f"Iniciando a simulação Flower [{rodada}]...")
        print("A saída completa será salva em 'simulation_log.txt'.")

        # Abre o arquivo de log para escrita
        with open("simulation_log.txt", "w") as log:
            subprocess.run(comando, stdout=log, stderr=subprocess.STDOUT)

        print(f"Simulação [{rodada}] concluída.")

        # Arquivo de saida com os parametros usados. Alguns tem que arredondar por algum motivo （；¬＿¬)
        plot_metrics_from_log("simulation_log.txt", round(fraction_fit, 1), local_epochs,batch_size, round(learning_rate, 3))


if __name__ == "__main__":

    num_testes = 10

    fraction_fit_list =  np.arange(0.1, 1.0, 0.1)
    local_epochs_list = random_values(1,20)
    learning_rate_list = np.arange(0.01, 0.000 , -0.001)
    batch_size_list = random_values(10,50)

    random_optimization(fraction_fit_list, local_epochs_list, learning_rate_list,batch_size_list, num_testes)








