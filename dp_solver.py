"""
Makespan Minimization — Solución exacta con Bitmask DP
Problema: asignar n trabajos a 2 máquinas paralelas minimizando el makespan.

Complejidad: O(2^n) tiempo y espacio.
El factor n aparece al enumerar todos los subconjuntos; aquí se optimiza
usando el bit menos significativo para calcular dp[mask] en O(1) por estado.
"""

import time
from typing import List, Tuple


def solve(jobs: List[int]) -> Tuple[int, List[List[int]], float]:
    """
    Bitmask DP para m=2 máquinas paralelas idénticas.

    Idea: dp[mask] = suma de tiempos de los trabajos en 'mask' (máquina 1).
          La máquina 2 toma todos los trabajos restantes → carga = total - dp[mask].
          Makespan = max(dp[mask], total - dp[mask]).
          Respuesta = min sobre todos los masks.

    Args:
        jobs: lista de tiempos de procesamiento (enteros positivos)
    Returns:
        (makespan_optimo, asignacion, tiempo_ejecucion_seg)
        asignacion[i] = índices de trabajos asignados a máquina i+1
    """
    n = len(jobs)
    if n == 0:
        return 0, [[], []], 0.0
    if n > 25:
        raise ValueError(
            f"n={n} excede el límite de 25 para Bitmask DP "
            f"(requeriría {2**n:,} estados)"
        )

    total = sum(jobs)
    start = time.perf_counter()

    # Precomputar dp[mask] usando recurrencia sobre el bit menos significativo:
    # dp[mask] = dp[mask ^ lsb] + jobs[indice_lsb]
    # Esto evita recalcular la suma completa para cada subconjunto.
    dp = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb_index = (mask & -mask).bit_length() - 1  # índice del bit menos significativo
        dp[mask] = dp[mask ^ (1 << lsb_index)] + jobs[lsb_index]

    best_makespan = total  # peor caso: todos los trabajos en máquina 1
    best_mask = 0
    for mask in range(1 << n):
        ms = max(dp[mask], total - dp[mask])
        if ms < best_makespan:
            best_makespan = ms
            best_mask = mask

    elapsed = time.perf_counter() - start

    machine1 = [i for i in range(n) if best_mask & (1 << i)]
    machine2 = [i for i in range(n) if not (best_mask & (1 << i))]
    return best_makespan, [machine1, machine2], elapsed


def makespan_only(jobs: List[int]) -> Tuple[int, float]:
    """
    Variante optimizada que solo retorna el makespan y el tiempo.
    Usada en benchmarks para evitar overhead de reconstrucción.
    """
    n = len(jobs)
    if n == 0:
        return 0, 0.0

    total = sum(jobs)
    start = time.perf_counter()

    dp = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb_index = (mask & -mask).bit_length() - 1
        dp[mask] = dp[mask ^ (1 << lsb_index)] + jobs[lsb_index]

    best = min(max(dp[m], total - dp[m]) for m in range(1 << n))

    return best, time.perf_counter() - start
