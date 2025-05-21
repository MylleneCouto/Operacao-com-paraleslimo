import subprocess
import matplotlib.pyplot as plt
import re
import time
import csv
import os
from colorama import Fore, Style, init

init(autoreset=True)

# Nome dos arquivos que serão executados
scripts_mpi = {
    "Original": "operacao_matriz_mpi.py",
    "Balanceado": "operacao_matriz_mpi_v2.py",
    "Balanceada e sem Gather": "operacao_matriz_mpi_v3.py",
    "Sem Scatter": "operacao_matriz_mpi_v4.py"
}

todos_processos = [1, 2, 4, 8]

pasta_relatorio = "relatorio"
os.makedirs(pasta_relatorio, exist_ok=True)

# Para armazenar todos os tempos
resultados = {}

for nome, arquivo in scripts_mpi.items():
    print(f"\n{Fore.CYAN}Testando variante: {Style.BRIGHT}{nome}")
    tempos = []

    subpasta = os.path.join(pasta_relatorio, nome.replace(" ", "_").lower())
    os.makedirs(subpasta, exist_ok=True)

    parar = False

    for i, p in enumerate(todos_processos):
        if parar:
            break

        total_tempo = 0.0
        execucoes = 3
        print(f"\n{Fore.YELLOW}Executando {arquivo} com {p} processo(s)...")

        for _ in range(execucoes):
            resultado = subprocess.run(
                ["mpirun", "--oversubscribe", "-np", str(p), "python3", arquivo],
                capture_output=True,
                text=True
            )

            match = re.search(r"Tempo.*?: ([\d.]+)", resultado.stdout)
            if match:
                tempo = float(match.group(1))
                total_tempo += tempo
            else:
                print(f"{Fore.RED}Saída sem tempo detectado:")
                print(resultado.stdout)

            time.sleep(0.5)

        tempo_medio = total_tempo / execucoes
        tempos.append(tempo_medio)
        print(f"{Fore.GREEN}Média de tempo: {tempo_medio:.4f} s")

        if i > 0:
            t1 = tempos[0]
            speedup = t1 / tempo_medio
            eficiencia = speedup / p
            print(f"{Fore.MAGENTA}Eficiência com {p} processos: {eficiencia:.2%}")
            if eficiencia < 0.3:
                print(f"{Fore.RED}Eficiência abaixo de 30%. Interrompendo testes para esta variante.")
                parar = True

    processos_usados = todos_processos[:len(tempos)]
    resultados[nome] = (processos_usados, tempos)

    # Gráficos por variante
    t1 = tempos[0]
    speedup = [t1 / t for t in tempos]
    eficiencia = [s / p for s, p in zip(speedup, processos_usados)]

    # Speedup
    plt.figure(figsize=(8, 5))
    plt.plot(processos_usados, speedup, marker='o', label='Speedup')
    plt.plot(processos_usados, processos_usados, linestyle='--', color='gray', label='Speedup Ideal')
    plt.xlabel('Número de processos (p)')
    plt.ylabel('Speedup S(p)')
    plt.title(f'Speedup - {nome}')
    plt.grid(True)
    plt.legend()
    plt.xticks(processos_usados)
    plt.savefig(os.path.join(subpasta, "speedup.png"))
    plt.close()

    # Eficiência
    plt.figure(figsize=(8, 5))
    plt.plot(processos_usados, eficiencia, marker='s', color='orange', label='Eficiência')
    plt.xlabel('Número de processos (p)')
    plt.ylabel('Eficiência E(p)')
    plt.title(f'Eficiência - {nome}')
    plt.grid(True)
    plt.legend()
    plt.xticks(processos_usados)
    plt.savefig(os.path.join(subpasta, "eficiencia.png"))
    plt.close()

# Salvar CSV final
csv_path = os.path.join(pasta_relatorio, "resultados_tempos.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Versão"] + [f"{p} proc" for p in todos_processos])
    for nome, (procs, tempos) in resultados.items():
        linha = [f"{t:.4f}" if p in procs else "" for p, t in zip(todos_processos, [tempos[i] if i < len(tempos) else None for i in range(len(todos_processos))])]
        writer.writerow([nome] + linha)

print(f"\n{Fore.GREEN}Relatório automático finalizado!")
print(f"{Fore.BLUE}Gráficos e tempos disponíveis em: {Style.BRIGHT}relatorio/")
