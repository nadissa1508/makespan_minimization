"""
Módulo de benchmarking empírico para Bitmask DP y LPT Greedy.
Genera las entradas de prueba, mide tiempos y compara calidad de soluciones.
"""

import random
import numpy as np
from typing import Dict, List, Optional

from dp_solver import makespan_only as _dp
from greedy_solver import makespan_only as _lpt


# ─────────────────────────────────────────────
# Generador de instancias reproducibles
# ─────────────────────────────────────────────

def generate_jobs(n: int, lo: int = 1, hi: int = 100, seed: int = 0) -> List[int]:
    """Genera n tiempos de procesamiento aleatorios en [lo, hi]."""
    return [random.Random(seed).randint(lo, hi) for _ in range(n)]


# ─────────────────────────────────────────────
# Benchmarks de tiempo de ejecución
# ─────────────────────────────────────────────

def run_dp_benchmark(
    sizes: Optional[List[int]] = None,
    trials: int = 5,
) -> Dict:
    """
    Mide el tiempo de Bitmask DP para distintos valores de n.
    n debe ser pequeño (≤ 20) por la complejidad O(2^n).

    Returns:
        dict con 'sizes', 'times' (promedio en segundos), 'std'
    """
    if sizes is None:
        sizes = [5, 8, 10, 12, 14, 16, 18, 20]

    results: Dict = {"sizes": [], "times": [], "std": []}
    print("\nBenchmark - Bitmask DP (m=2):")
    print(f"  {'n':>5}  {'promedio (s)':>14}  {'std (s)':>12}  {'promedio (ms)':>14}")
    print("  " + "-" * 52)

    for n in sizes:
        ts = [_dp(generate_jobs(n, seed=t * 997 + n))[1] for t in range(trials)]
        avg, std = float(np.mean(ts)), float(np.std(ts))
        results["sizes"].append(n)
        results["times"].append(avg)
        results["std"].append(std)
        print(f"  {n:>5}  {avg:>14.6f}  {std:>12.6f}  {avg*1000:>13.4f}ms")

    return results


def run_lpt_benchmark(
    sizes: Optional[List[int]] = None,
    trials: int = 10,
    m: int = 2,
) -> Dict:
    """
    Mide el tiempo de LPT Greedy para distintos valores de n.
    Soporta n muy grande gracias a su complejidad O(n log n).

    Returns:
        dict con 'sizes', 'times' (promedio en segundos), 'std'
    """
    if sizes is None:
        sizes = [10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000]

    results: Dict = {"sizes": [], "times": [], "std": []}
    print(f"\nBenchmark - LPT Greedy (m={m}):")
    print(f"  {'n':>8}  {'promedio (s)':>14}  {'std (s)':>12}  {'promedio (ms)':>14}")
    print("  " + "-" * 56)

    for n in sizes:
        ts = [_lpt(generate_jobs(n, seed=t * 997 + n), m)[1] for t in range(trials)]
        avg, std = float(np.mean(ts)), float(np.std(ts))
        results["sizes"].append(n)
        results["times"].append(avg)
        results["std"].append(std)
        print(f"  {n:>8}  {avg:>14.6f}  {std:>12.6f}  {avg*1000:>13.4f}ms")

    return results