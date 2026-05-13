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


# ─────────────────────────────────────────────
# Comparación de calidad de solución
# ─────────────────────────────────────────────

def run_quality_comparison(
    sizes: Optional[List[int]] = None,
    trials: int = 30,
    m: int = 2,
) -> Dict:
    """
    Compara el makespan óptimo (DP) vs el makespan greedy (LPT)
    para instancias pequeñas donde el DP es factible.

    Returns:
        dict con 'sizes', 'dp_ms', 'lpt_ms', 'ratios'
    """
    if sizes is None:
        sizes = list(range(3, 21))

    results: Dict = {
        "sizes":   [],
        "dp_ms":   [],
        "lpt_ms":  [],
        "ratios":  [],
        "dp_times": [],
        "lpt_times": [],
    }

    print(f"\nComparacion de Calidad - DP optimo vs LPT Greedy (m={m}):")
    print(f"  {'n':>4}  {'DP (avg)':>10}  {'LPT (avg)':>10}  {'ratio':>8}  {'gap%':>7}")
    print("  " + "-" * 48)

    for n in sizes:
        dp_list, lpt_list, dp_t_list, lpt_t_list = [], [], [], []
        for t in range(trials):
            jobs = generate_jobs(n, seed=t * 997 + n)
            dp_val, dp_t   = _dp(jobs)
            lpt_val, lpt_t = _lpt(jobs, m)
            dp_list.append(dp_val)
            lpt_list.append(lpt_val)
            dp_t_list.append(dp_t)
            lpt_t_list.append(lpt_t)

        avg_dp  = float(np.mean(dp_list))
        avg_lpt = float(np.mean(lpt_list))
        ratio   = avg_lpt / avg_dp if avg_dp > 0 else 1.0
        gap_pct = (ratio - 1) * 100

        results["sizes"].append(n)
        results["dp_ms"].append(avg_dp)
        results["lpt_ms"].append(avg_lpt)
        results["ratios"].append(ratio)
        results["dp_times"].append(float(np.mean(dp_t_list)))
        results["lpt_times"].append(float(np.mean(lpt_t_list)))

        print(f"  {n:>4}  {avg_dp:>10.2f}  {avg_lpt:>10.2f}  {ratio:>8.4f}  {gap_pct:>6.2f}%")

    return results


# ─────────────────────────────────────────────
# Instancias adversariales para LPT (m=2)
# ─────────────────────────────────────────────

def adversarial_instances_m2() -> List[tuple]:
    """
    Genera instancias donde LPT es SUBOPTIMO para m=2.

    Construccion: jobs = [a+1, a+1, a, a, a-1]  (n=5, m=2)
    - LPT makespan = 3a
    - Optimo makespan = 3a-1  (M1=[a+1,a+1], M2=[a,a,a-1])
    - Ratio = 3a / (3a-1) > 1  para todo a >= 2
    """
    instances = []
    for a in range(2, 16):
        jobs = [a + 1, a + 1, a, a, a - 1]
        instances.append((f"adv_a={a:02d}", jobs, 2))

    # Instancias aleatorias con rango pequeno [1,8]
    rng = random.Random(999)
    for i in range(10):
        n = rng.randint(5, 18)
        jobs = [rng.randint(1, 8) for _ in range(n)]
        instances.append((f"rand_small_{i:02d}", jobs, 2))

    return instances


def run_adversarial_comparison() -> List[dict]:
    """
    Corre DP vs LPT en instancias adversariales.
    Muestra casos donde LPT no alcanza el optimo.
    """
    from dp_solver import solve as dp_solve
    from greedy_solver import solve as lpt_solve

    instances = adversarial_instances_m2()
    results = []
    suboptimal_count = 0

    print("\nInstancias adversariales para LPT (m=2):")
    print(f"  {'instancia':<18}  {'n':>3}  {'DP':>5}  {'LPT':>5}  {'ratio':>8}")
    print("  " + "-" * 50)

    for name, jobs, m in instances:
        dp_ms, _, _  = dp_solve(jobs)
        lpt_ms, _, _ = lpt_solve(jobs, m)
        ratio = lpt_ms / dp_ms if dp_ms > 0 else 1.0
        flag = "  <- SUBOPTIMO" if ratio > 1.0 else ""
        print(f"  {name:<18}  {len(jobs):>3}  {dp_ms:>5}  {lpt_ms:>5}  {ratio:>8.4f}{flag}")
        if ratio > 1.0:
            suboptimal_count += 1
        results.append({
            "name": name, "jobs": jobs, "m": m,
            "dp_ms": dp_ms, "lpt_ms": lpt_ms, "ratio": ratio,
        })

    print(f"\n  Instancias suboptimas: {suboptimal_count} / {len(instances)}")
    return results