"""
Makespan Minimization — Menú Principal
Proyecto #2 | Análisis y Diseño de Algoritmos

Integrantes:
  - Javier Linares   (231135)
  - Gadiel Ocaña     (231270)
  - Angie Vela       (23764)

"""

import os
import sys
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dp_solver import solve as dp_solve
from greedy_solver import solve as lpt_solve
from benchmark import (
    run_dp_benchmark, run_lpt_benchmark,
    run_quality_comparison, run_adversarial_comparison,
)
from visualizer import (
    plot_timing_comparison,
    plot_quality_comparison,
    plot_gantt,
    plot_dp_log_scale,
)


# Constantes de presentación

BANNER = """
+==================================================================+
|      MAKESPAN MINIMIZATION EN MAQUINAS PARALELAS                |
|      Proyecto #2 - Analisis y Diseno de Algoritmos              |
|      Universidad del Valle de Guatemala  -  2026                |
+==================================================================+
|  Javier Linares (231135)  -  Gadiel Ocana (231270)              |
|  Angie Vela (23764)                                             |
+==================================================================+
|  Algoritmos:                                                    |
|    * Bitmask DP    - Solucion EXACTA    O(2^n),   m = 2        |
|    * LPT Greedy    - HEURISTICA         O(n log n), m >= 2     |
+==================================================================+
"""

SEP = "-" * 66


# Utilidades de presentación

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _menu():
    print(f"\n{SEP}")
    print("  MENU PRINCIPAL")
    print(SEP)
    print("  1.  Resolver problema personalizado")
    print("  2.  Ejecutar benchmark completo  ->  genera graficas PNG")
    print("  3.  Comparar calidad de soluciones (DP vs LPT, instancias aleatorias)")
    print("  4.  Ver diagrama de Gantt (ejemplo interactivo)")
    print("  5.  Demo rapido  (n = 10, m = 2)")
    print("  6.  Analisis de escala logaritmica DP")
    print("  7.  Demo instancias ADVERSARIALES (muestra gap LPT vs optimo)")
    print("  8.  Salir")
    print(SEP)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    if default:
        prompt = f"{prompt} [{default}]: "
    val = input(prompt).strip()
    return val if val else (default or "")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = int(raw)
            if v >= 1:
                return v
        except ValueError:
            pass
        print("  Ingrese un entero positivo.")


# Impresión de resultados

def _print_solution(
    jobs: List[int],
    dp_result: Tuple,
    lpt_result: Tuple,
    m: int,
):
    dp_ms, dp_assign, dp_time = dp_result
    lpt_ms, lpt_assign, lpt_time = lpt_result

    print(f"\n{'='*66}")
    print(f"  RESULTADOS")
    print(f"{'='*66}")
    print(f"  Trabajos (n={len(jobs)}):  {jobs}")
    print(f"  Maquinas (m):  {m}")
    print(f"  Suma total de tiempos:  {sum(jobs)}")

    print(f"\n  {'-'*30} Bitmask DP (Exacto) {'-'*14}")
    print(f"  Makespan optimo   :  {dp_ms}")
    print(f"  Tiempo ejecucion  :  {dp_time * 1000:.4f} ms")
    for i, mach in enumerate(dp_assign):
        times = [jobs[j] for j in mach]
        label = [j + 1 for j in mach]
        print(f"  Maquina {i+1}  ->  trabajos {label}  |  tiempos {times}  |  suma = {sum(times)}")

    print(f"\n  {'-'*30} LPT Greedy {'-'*22}")
    print(f"  Makespan LPT      :  {lpt_ms}")
    print(f"  Tiempo ejecucion  :  {lpt_time * 1000:.4f} ms")
    for i, mach in enumerate(lpt_assign):
        times = [jobs[j] for j in mach]
        label = [j + 1 for j in mach]
        print(f"  Maquina {i+1}  ->  trabajos {label}  |  tiempos {times}  |  suma = {sum(times)}")

    ratio = lpt_ms / dp_ms if dp_ms > 0 else 1.0
    gap   = (ratio - 1) * 100
    print(f"\n  {'-'*30} Comparacion {'-'*20}")
    print(f"  Ratio LPT / DP optimo  :  {ratio:.6f}")
    if ratio == 1.0:
        print("  LPT encontro la solucion optima en este caso.")
    else:
        print(f"  LPT esta {gap:.2f}% por encima del optimo.")
    if lpt_time > 0:
        print(f"  Aceleracion LPT vs DP  :  {dp_time / lpt_time:.1f}x mas rapido")
    print(f"{'='*66}")


# Opción 1: Problema personalizado

def _option_custom():
    print(f"\n{SEP}")
    print("  PROBLEMA PERSONALIZADO")
    print(SEP)

    print("  Ingrese los tiempos de procesamiento como enteros positivos.")
    print("  Ejemplo:  3 7 2 5 8 1  (separados por espacios)")

    while True:
        raw = input("\n  Tiempos: ").strip()
        try:
            jobs = list(map(int, raw.split()))
            if not jobs or any(j <= 0 for j in jobs):
                print("  Todos los tiempos deben ser enteros positivos.")
                continue
            break
        except ValueError:
            print("  Formato inválido. Use enteros separados por espacios.")

    m = _ask_int("  Numero de maquinas (m)", default=2)
    n = len(jobs)

    if m > 2:
        print(f"\n  Nota: Bitmask DP solo funciona con m=2 maquinas.")
        print(f"  El DP se ejecutara con m=2; LPT usara m={m}.")

    run_dp = True
    if n > 20:
        print(f"\n  Advertencia: n={n} -> DP requiere 2^{n} = {2**n:,} estados.")
        print("  Puede tardar mucho tiempo o agotar memoria.")
        run_dp = _ask("  Ejecutar DP de todas formas? (s/n)", "n").lower() == "s"

    if run_dp:
        try:
            dp_result = dp_solve(jobs)          # siempre m=2
        except ValueError as e:
            print(f"  Error DP: {e}")
            run_dp = False

    lpt_result = lpt_solve(jobs, m)             # m especificado por usuario

    if run_dp:
        _print_solution(jobs, dp_result, lpt_result, m)
        gantt = _ask("\n  Mostrar diagrama de Gantt? (s/n)", "s").lower() == "s"
        if gantt:
            # Gantt DP siempre con 2 maquinas; LPT con m del usuario
            fig = plot_gantt(jobs, dp_result[1], lpt_result[1])
            plt.show()
    else:
        print(f"\n  LPT Makespan : {lpt_result[0]}")
        print(f"  Tiempo LPT   : {lpt_result[2]*1000:.4f} ms")
        for i, mach in enumerate(lpt_result[1]):
            times = [jobs[j] for j in mach]
            print(f"  Maquina {i+1}  ->  trabajos {[j+1 for j in mach]}  |  suma = {sum(times)}")


# Opción 2: Benchmark completo

def _option_benchmark():
    print(f"\n{SEP}")
    print("  BENCHMARK COMPLETO")
    print(SEP)
    print("  Tamanos DP  : n in {5, 8, 10, 12, 14, 16, 18, 20}")
    print("  Tamanos LPT : n in {10, 50, 100, 500, 1k, 5k, 10k, 50k, 100k}")
    print("  Esto puede tardar 2-5 minutos.\n")

    dp_results  = run_dp_benchmark()
    lpt_results = run_lpt_benchmark()

    os.makedirs("graficas", exist_ok=True)
    fig1 = plot_timing_comparison(dp_results, lpt_results,
                                  save_path="graficas/tiempos_comparacion.png")
    fig2 = plot_dp_log_scale(dp_results,
                             save_path="graficas/dp_escala_log.png")

    print("\n  Gráficas guardadas en carpeta 'graficas/'.")
    print("  Mostrando figuras...")
    plt.show()

# Opción 3: Comparación de calidad

def _option_quality():
    print(f"\n{SEP}")
    print("  COMPARACION DE CALIDAD DE SOLUCIONES")
    print(SEP)
    print("  Se ejecutan 30 instancias aleatorias por cada n (n = 3..20).\n")

    quality = run_quality_comparison()

    os.makedirs("graficas", exist_ok=True)
    fig = plot_quality_comparison(quality, save_path="graficas/calidad_solucion.png")

    print("\n  Gráfica guardada en 'graficas/calidad_solucion.png'.")
    plt.show()

# Opción 4: Gantt interactivo

def _option_gantt():
    print(f"\n{SEP}")
    print("  DIAGRAMA DE GANTT INTERACTIVO")
    print(SEP)

    print("  Opciones de ejemplo:")
    print("    a) Ejemplo pequeno  (n=6,  m=2, predefinido)")
    print("    b) Ejemplo mediano  (n=10, m=2, aleatorio)")
    print("    c) Mas maquinas     (n=12, m=3, aleatorio)")
    print("    d) Personalizado")

    choice = _ask("  Seleccione (a/b/c/d)", "b").lower()

    if choice == "a":
        jobs = [8, 3, 7, 5, 2, 6]
        m = 2
    elif choice == "b":
        random.seed(42)
        jobs = [random.randint(1, 20) for _ in range(10)]
        m = 2
    elif choice == "c":
        random.seed(7)
        jobs = [random.randint(1, 15) for _ in range(12)]
        m = 3
    else:
        raw = input("  Tiempos (enteros separados por espacios): ").strip()
        try:
            jobs = list(map(int, raw.split()))
        except ValueError:
            print("  Formato inválido, usando ejemplo por defecto.")
            jobs = [8, 3, 7, 5, 2, 6]
        m = _ask_int("  Número de máquinas (m)", 2)

    print(f"\n  Trabajos : {jobs}")
    print(f"  Máquinas : {m}")

    if m > 2:
        print(f"\n  Nota: Bitmask DP solo opera con m=2. LPT usara m={m}.")

    if len(jobs) > 20:
        print("  Advertencia: n > 20. Usando LPT para el subplot de DP tambien.")
        dp_assign = lpt_solve(jobs, 2)[1]   # LPT con 2 maquinas como proxy
    else:
        dp_assign = dp_solve(jobs)[1]       # DP exacto, siempre m=2

    lpt_assign = lpt_solve(jobs, m)[1]      # LPT con m maquinas del usuario

    save = _ask("  Guardar diagrama? (s/n)", "n").lower() == "s"
    path = "graficas/gantt.png" if save else None
    if save:
        os.makedirs("graficas", exist_ok=True)

    # plot_gantt usa len(assignment) por subplot, soporta m distinto en cada uno
    fig = plot_gantt(jobs, dp_assign, lpt_assign, save_path=path)
    plt.show()


# Opción 5: Demo rápido

def _option_demo():
    print(f"\n{SEP}")
    print("  DEMO RAPIDO - n=10, m=2")
    print(SEP)

    random.seed(42)
    jobs = [random.randint(1, 20) for _ in range(10)]
    m = 2
    print(f"  Trabajos generados (seed=42):  {jobs}")

    dp_result  = dp_solve(jobs)
    lpt_result = lpt_solve(jobs, m)

    _print_solution(jobs, dp_result, lpt_result, m)

    gantt = _ask("\n  Mostrar diagrama de Gantt? (s/n)", "s").lower() == "s"
    if gantt:
        fig = plot_gantt(jobs, dp_result[1], lpt_result[1])
        plt.show()


# Opción 6: Escala logarítmica DP

def _option_log_scale():
    print(f"\n{SEP}")
    print("  ANALISIS ESCALA LOGARITMICA - DP")
    print(SEP)
    print("  Ejecuta benchmark DP y muestra crecimiento en escala semilog.")
    print("  Confirma visualmente la complejidad O(2^n).\n")

    dp_results = run_dp_benchmark()

    os.makedirs("graficas", exist_ok=True)
    fig = plot_dp_log_scale(dp_results, save_path="graficas/dp_escala_log.png")
    print("\n  Guardado en 'graficas/dp_escala_log.png'.")
    plt.show()


# Opcion 7: Instancias adversariales

def _option_adversarial():
    print(f"\n{SEP}")
    print("  INSTANCIAS ADVERSARIALES - LPT vs DP (m=2)")
    print(SEP)
    print("  Construccion: jobs = [a+1, a+1, a, a, a-1]")
    print("  LPT makespan = 3a   vs   Optimo = 3a-1")
    print("  Ratio = 3a/(3a-1) > 1  (LPT es suboptimo en estos casos)\n")

    results = run_adversarial_comparison()

    # Grafica de ratios para instancias adversariales
    adv_results = [r for r in results if r["name"].startswith("adv_")]
    if adv_results:
        import matplotlib
        matplotlib.use("TkAgg")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Instancias Adversariales: LPT vs DP (m=2)", fontsize=13, fontweight="bold")

        names_short = [r["name"].replace("adv_a=", "a=") for r in adv_results]
        dp_vals  = [r["dp_ms"]  for r in adv_results]
        lpt_vals = [r["lpt_ms"] for r in adv_results]
        ratios   = [r["ratio"]  for r in adv_results]
        x = np.arange(len(names_short))

        # Barras comparativas
        w = 0.35
        axes[0].bar(x - w/2, dp_vals,  w, label="DP (optimo)", color="#e74c3c", alpha=0.85)
        axes[0].bar(x + w/2, lpt_vals, w, label="LPT",         color="#2980b9", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names_short, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("Makespan", fontsize=11)
        axes[0].set_title("Makespan DP vs LPT", fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, axis="y", alpha=0.3)

        # Ratio de aproximacion
        a_vals = list(range(2, 2 + len(adv_results)))
        teorico = [3*a / (3*a - 1) for a in a_vals]
        axes[1].plot(a_vals, ratios,  "o-", color="#8e44ad", lw=2, ms=7, label="Ratio medido")
        axes[1].plot(a_vals, teorico, "s--", color="#e67e22", lw=2, ms=5, label="Ratio teorico 3a/(3a-1)")
        axes[1].axhline(y=1.0, color="#27ae60", ls="-", lw=1.5, label="Optimo (ratio=1)")
        axes[1].set_xlabel("Parametro a", fontsize=11)
        axes[1].set_ylabel("Ratio LPT / DP", fontsize=11)
        axes[1].set_title("Ratio de Aproximacion", fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.97, 1.15)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save = _ask("\n  Guardar grafica? (s/n)", "s").lower() == "s"
        if save:
            os.makedirs("graficas", exist_ok=True)
            plt.savefig("graficas/adversarial.png", dpi=150, bbox_inches="tight")
            print("  Guardado: graficas/adversarial.png")
        plt.show()


# Main loop

def main():
    print(BANNER)

    actions = {
        "1": _option_custom,
        "2": _option_benchmark,
        "3": _option_quality,
        "4": _option_gantt,
        "5": _option_demo,
        "6": _option_log_scale,
        "7": _option_adversarial,
    }

    while True:
        _menu()
        choice = input("  Seleccione una opcion (1-8): ").strip()

        if choice == "8":
            print("\n  !Hasta luego!\n")
            break
        elif choice in actions:
            try:
                actions[choice]()
            except KeyboardInterrupt:
                print("\n  Operacion cancelada.")
        else:
            print("  Opcion invalida. Ingrese un numero del 1 al 8.")

        input("\n  Presione Enter para continuar...")


if __name__ == "__main__":
    main()