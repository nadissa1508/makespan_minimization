"""
Makespan Minimization — Heurística Greedy LPT
Algoritmo: Longest Processing Time First (LPT)

Complejidad: O(n log n) por ordenamiento + O(n log m) por operaciones de heap.
             Total: O(n log n) dado que m << n en la práctica.

Ratio de aproximación garantizado: makespan_LPT / makespan_OPT ≤ 4/3 - 1/(3m)
"""

import heapq
import time
from typing import List, Tuple


def solve(jobs: List[int], m: int = 2) -> Tuple[int, List[List[int]], float]:
    """
    LPT Greedy para m máquinas paralelas idénticas.

    Criterio greedy: siempre asignar el trabajo más largo disponible
                     a la máquina con menor carga actual.

    Args:
        jobs: lista de tiempos de procesamiento
        m:    número de máquinas (≥ 1)
    Returns:
        (makespan, asignacion, tiempo_ejecucion_seg)
        asignacion[i] = índices de trabajos asignados a máquina i+1
    """
    n = len(jobs)
    if n == 0:
        return 0, [[] for _ in range(m)], 0.0

    start = time.perf_counter()

    # Ordenar trabajos de mayor a menor, conservando índice original
    sorted_jobs = sorted(enumerate(jobs), key=lambda x: -x[1])

    # Min-heap: (carga_actual, id_maquina)
    # Invariante: la raíz siempre es la máquina menos cargada
    heap = [(0, i) for i in range(m)]
    heapq.heapify(heap)

    assignment = [[] for _ in range(m)]
    loads = [0] * m

    for orig_idx, proc_time in sorted_jobs:
        load, mid = heapq.heappop(heap)
        assignment[mid].append(orig_idx)
        loads[mid] = load + proc_time
        heapq.heappush(heap, (loads[mid], mid))

    makespan = max(loads)
    elapsed = time.perf_counter() - start
    return makespan, assignment, elapsed


def makespan_only(jobs: List[int], m: int = 2) -> Tuple[int, float]:
    """
    Variante optimizada que solo retorna el makespan y el tiempo.
    Usada en benchmarks.
    """
    n = len(jobs)
    if n == 0:
        return 0, 0.0

    start = time.perf_counter()

    heap = [0] * m
    heapq.heapify(heap)
    for t in sorted(jobs, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + t)

    makespan = max(heap)
    return makespan, time.perf_counter() - start
