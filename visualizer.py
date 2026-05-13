"""
Módulo de visualización para el análisis empírico del Proyecto #2.

Genera:
  - Scatter plots con regresión polinomial (tiempos de ejecución)
  - Diagrama de Gantt (asignación de trabajos a máquinas)
  - Gráfica de calidad de solución (makespan DP vs LPT)
  - Gráfica de ratio de aproximación
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional


# Helpers de regresión

def _best_poly_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_deg: int = 6,
) -> Tuple[np.ndarray, int, float]:
    """
    Elige el grado de regresión polinomial con mayor R² ajustado.
    Retorna (coeficientes, grado, R²_ajustado).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    best_deg, best_r2, best_coeffs = 1, -np.inf, np.array([0.0, 0.0])
    for deg in range(1, min(max_deg + 1, n)):
        try:
            coeffs = np.polyfit(x, y, deg)
        except np.RankWarning:
            continue
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        denom = n - deg - 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / denom if denom > 0 else r2
        if adj_r2 > best_r2:
            best_r2, best_deg, best_coeffs = adj_r2, deg, coeffs

    return best_coeffs, best_deg, best_r2


def _poly_label(coeffs: np.ndarray, deg: int) -> str:
    """Formato legible para el polinomio."""
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-15:
            continue
        coef_str = f"{c:.3e}"
        if power == 0:
            terms.append(coef_str)
        elif power == 1:
            terms.append(f"{coef_str}·n")
        else:
            terms.append(f"{coef_str}·n^{power}")
    return " + ".join(terms) if terms else "0"


# Plot 1: Tiempos de ejecución — DP

def plot_dp_benchmark(results: Dict, ax=None, save_path: Optional[str] = None):
    x = np.array(results["sizes"], dtype=float)
    y_ms = np.array(results["times"], dtype=float) * 1000   # → ms
    std_ms = np.array(results["std"], dtype=float) * 1000

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(x, y_ms, yerr=std_ms, fmt="o", color="#e74c3c", ms=8,
                capsize=5, capthick=1.5, elinewidth=1.2,
                label="Tiempo medido (promedio ± std)", zorder=3)

    coeffs, deg, r2 = _best_poly_fit(x, y_ms, max_deg=6)
    x_fine = np.linspace(x.min(), x.max(), 300)
    y_fit = np.polyval(coeffs, x_fine)
    ax.plot(x_fine, y_fit, "--", color="#922b21", lw=2.2,
            label=f"Regresión polinomial grado {deg}  (R²={r2:.4f})")

    ax.set_xlabel("Número de trabajos (n)", fontsize=12)
    ax.set_ylabel("Tiempo de ejecución (ms)", fontsize=12)
    ax.set_title("Bitmask DP — Tiempo de ejecución vs n\n"
                 "Complejidad O(2ⁿ) — m = 2 máquinas", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(results["sizes"])
    ax.tick_params(axis="x", rotation=45)

    # Anotar puntos
    for xi, yi in zip(x, y_ms):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#922b21")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Guardado: {save_path}")
        return fig


# Plot 2: Tiempos de ejecución — LPT

def plot_lpt_benchmark(results: Dict, ax=None, save_path: Optional[str] = None):
    x = np.array(results["sizes"], dtype=float)
    y_ms = np.array(results["times"], dtype=float) * 1000
    std_ms = np.array(results["std"], dtype=float) * 1000

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(x, y_ms, yerr=std_ms, fmt="s", color="#2980b9", ms=8,
                capsize=5, capthick=1.5, elinewidth=1.2,
                label="Tiempo medido (promedio ± std)", zorder=3)

    coeffs, deg, r2 = _best_poly_fit(x, y_ms, max_deg=3)
    x_fine = np.linspace(x.min(), x.max(), 300)
    y_fit = np.polyval(coeffs, x_fine)
    ax.plot(x_fine, y_fit, "--", color="#1a5276", lw=2.2,
            label=f"Regresión polinomial grado {deg}  (R²={r2:.4f})")

    ax.set_xlabel("Número de trabajos (n)", fontsize=12)
    ax.set_ylabel("Tiempo de ejecución (ms)", fontsize=12)
    ax.set_title("LPT Greedy — Tiempo de ejecución vs n\n"
                 "Complejidad O(n log n)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Guardado: {save_path}")
        return fig

# Plot 3: Comparativa DP vs LPT (tiempos)

def plot_timing_comparison(
    dp_results: Dict,
    lpt_results: Dict,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(
        "Análisis Empírico — Tiempo de Ejecución\n"
        "Makespan Minimization: Bitmask DP vs LPT Greedy",
        fontsize=14, fontweight="bold",
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot_dp_benchmark(dp_results, ax=ax1)
    plot_lpt_benchmark(lpt_results, ax=ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {save_path}")
    return fig


# Plot 4: Calidad de solución DP vs LPT

def plot_quality_comparison(
    results: Dict,
    save_path: Optional[str] = None,
):
    x   = np.array(results["sizes"])
    dp  = np.array(results["dp_ms"])
    lpt = np.array(results["lpt_ms"])
    ratios = np.array(results["ratios"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Calidad de Solución: DP Exacto vs LPT Greedy (m = 2 máquinas)",
        fontsize=13, fontweight="bold",
    )

    # — Makespan promedio —
    ax1.plot(x, dp,  "o-", color="#e74c3c", lw=2, ms=7, label="DP Bitmask (óptimo)")
    ax1.plot(x, lpt, "s--", color="#2980b9", lw=2, ms=7, label="LPT Greedy")
    ax1.fill_between(x, dp, lpt, alpha=0.12, color="#f39c12", label="Gap de calidad")
    ax1.set_xlabel("Número de trabajos (n)", fontsize=12)
    ax1.set_ylabel("Makespan promedio", fontsize=12)
    ax1.set_title("Makespan promedio vs n", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticks(x[::2])

    # — Ratio de aproximación —
    ax2.plot(x, ratios, "D-", color="#8e44ad", lw=2, ms=7, label="Ratio LPT / DP")
    # Cota LPT para m=2: 4/3 - 1/(3*2) = 7/6
    lpt_bound_m2 = 4/3 - 1/6  # = 7/6 ≈ 1.1667
    ax2.axhline(y=1.0, color="#27ae60", ls="-",  lw=1.8,
                label="Óptimo (ratio = 1.00)")
    ax2.axhline(y=lpt_bound_m2, color="#e67e22", ls="--", lw=1.8,
                label=f"Cota teórica LPT (m=2) = 7/6 ≈ {lpt_bound_m2:.4f}")
    ax2.fill_between(x, 1.0, ratios, alpha=0.15, color="#8e44ad")
    ax2.set_xlabel("Número de trabajos (n)", fontsize=12)
    ax2.set_ylabel("Ratio LPT / Óptimo", fontsize=12)
    ax2.set_title("Ratio de Aproximación LPT vs DP", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim(0.97, 1.25)
    ax2.set_xticks(x[::2])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {save_path}")
    return fig


# Plot 5: Diagrama de Gantt

def plot_gantt(
    jobs: List[int],
    dp_assignment: List[List[int]],
    lpt_assignment: List[List[int]],
    m: int = None,      # ignorado; cada subplot usa len(assignment)
    save_path: Optional[str] = None,
):
    """
    Genera un diagrama de Gantt comparando la asignación de DP vs LPT.
    Cada subplot usa el número de máquinas de su propia asignación,
    por lo que DP (m=2) y LPT (m cualquiera) se muestran correctamente.
    """
    n = len(jobs)
    cmap = plt.cm.get_cmap("tab20", n)
    colors = [cmap(i) for i in range(n)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    fig.suptitle(
        "Diagrama de Gantt — Asignación de Trabajos a Máquinas\n"
        "Bitmask DP (óptimo) vs LPT Greedy",
        fontsize=13, fontweight="bold",
    )

    configs = [
        (axes[0], dp_assignment,  "Bitmask DP  —  Solución Óptima", "#e74c3c"),
        (axes[1], lpt_assignment, "LPT Greedy  —  Heurística",       "#2980b9"),
    ]

    for ax, assignment, title, bar_edge in configs:
        # Usar la cantidad real de máquinas en esta asignación (DP siempre da 2)
        actual_m = len(assignment)
        machine_loads = [sum(jobs[j] for j in mach) for mach in assignment]
        makespan = max(machine_loads) if machine_loads else 0

        for mid, machine_jobs in enumerate(assignment):
            x_start = 0
            # Ordenar por tiempo descendente para visualización consistente
            for job_idx in sorted(machine_jobs, key=lambda j: jobs[j], reverse=True):
                duration = jobs[job_idx]
                ax.barh(
                    mid, duration, left=x_start, height=0.55,
                    color=colors[job_idx], edgecolor="black", lw=0.8,
                )
                # Etiqueta interna si hay espacio
                if duration >= max(jobs) * 0.07:
                    ax.text(
                        x_start + duration / 2, mid,
                        f"J{job_idx + 1}\n({duration})",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="black",
                    )
                x_start += duration

        ax.axvline(x=makespan, color="red", ls="--", lw=2,
                   label=f"Makespan = {makespan}")
        ax.set_yticks(range(actual_m))
        ax.set_yticklabels([f"Máquina {i + 1}" for i in range(actual_m)], fontsize=10)
        ax.set_xlabel("Tiempo de procesamiento", fontsize=11)
        ax.set_title(f"{title}    [Makespan = {makespan}]",
                     fontsize=11, fontweight="bold", color=bar_edge)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.set_xlim(0, None)

    # Leyenda global de trabajos (independiente de m)
    job_patches = [
        mpatches.Patch(color=colors[i], label=f"J{i+1}  (t={jobs[i]})")
        for i in range(n)
    ]
    ncols = min(n, 8)
    fig.legend(
        handles=job_patches,
        loc="lower center",
        ncol=ncols,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
        title="Trabajos",
        title_fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {save_path}")
    return fig


# Plot 6: Escala logarítmica DP (muestra crecimiento exponencial)

def plot_dp_log_scale(results: Dict, save_path: Optional[str] = None):
    x = np.array(results["sizes"], dtype=float)
    y_ms = np.array(results["times"], dtype=float) * 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Bitmask DP — Crecimiento Exponencial", fontsize=13, fontweight="bold")

    # Escala lineal
    ax1.plot(x, y_ms, "o-", color="#e74c3c", lw=2, ms=8, label="Tiempo (ms)")
    coeffs, deg, r2 = _best_poly_fit(x, y_ms, max_deg=7)
    x_fine = np.linspace(x.min(), x.max(), 300)
    ax1.plot(x_fine, np.polyval(coeffs, x_fine), "--", color="#922b21", lw=2,
             label=f"Regresión polinomial grado {deg} (R²={r2:.4f})")
    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("Tiempo (ms)", fontsize=12)
    ax1.set_title("Escala lineal", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticks(results["sizes"])

    # Escala log — ajuste exponencial
    log_y = np.log(y_ms + 1e-12)
    coeffs_log = np.polyfit(x, log_y, 1)  # log(y) ≈ a*n + b → y ≈ e^b * e^(a*n)
    r2_log_num = 1 - np.sum((log_y - np.polyval(coeffs_log, x))**2) / \
                     np.sum((log_y - log_y.mean())**2)

    ax2.semilogy(x, y_ms, "o", color="#e74c3c", ms=8, label="Tiempo (ms)")
    ax2.semilogy(x_fine, np.exp(np.polyval(coeffs_log, x_fine)), "--",
                 color="#922b21", lw=2,
                 label=f"Regresión exponencial  (R²={r2_log_num:.4f})\n"
                       f"y ≈ e^({coeffs_log[1]:.2f}) · e^({coeffs_log[0]:.4f}·n)")
    ax2.set_xlabel("n", fontsize=12)
    ax2.set_ylabel("Tiempo (ms) — escala log", fontsize=12)
    ax2.set_title("Escala semilogarítmica", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--", which="both")
    ax2.set_xticks(results["sizes"])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {save_path}")
    return fig
