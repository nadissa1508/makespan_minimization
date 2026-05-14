# Makespan Minimization — Proyecto #2
### Análisis y Diseño de Algoritmos | Universidad del Valle de Guatemala | Semestre 1, 2026

---

## Integrantes

| Nombre | Carné |
|--------|-------|
| Angie Vela | 23764 |
| Gadiel Ocaña | 231270 |
| Javier Linares | 231135 |

---

## Descripción del problema

El problema de **Makespan Minimization** consiste en asignar `n` trabajos con tiempos de procesamiento conocidos a `m` máquinas paralelas idénticas, de forma que el tiempo máximo de finalización (makespan) sea **mínimo**.

Este problema tiene aplicaciones en manufactura, computación paralela y distribución de tareas en servidores.

---

## Algoritmos implementados

### Bitmask DP — Solución exacta
- Complejidad: `O(2^n)`
- Limitado a `m = 2` máquinas
- Explora todos los subconjuntos posibles de asignación
- Garantiza la solución óptima

### LPT Greedy — Heurística
- Complejidad: `O(n log n)`
- Soporta cualquier número de máquinas `m ≥ 1`
- Ordena los trabajos de mayor a menor y asigna cada uno a la máquina con menor carga
- Ratio de aproximación garantizado: `makespan_LPT / makespan_OPT ≤ 4/3 - 1/(3m)`

---

## Estructura del proyecto

```
makespan_minimization/
│
├── greedy_solver.py   # Algoritmo LPT Greedy con cola de prioridad (heapq)
├── dp_solver.py       # Algoritmo Bitmask DP exacto para m=2 máquinas
├── benchmark.py       # Medición de tiempos y comparación de calidad de soluciones
├── visualizer.py      # Gráficas de dispersión, regresión polinomial y diagrama de Gantt
├── main.py            # Menú principal con todas las opciones de análisis
└── graficas/          # Carpeta generada automáticamente con las imágenes PNG
```

---

## Requisitos

- Python 3.8 o superior
- Instalar dependencias:

```
pip install matplotlib numpy
```

---

## Cómo ejecutar

```
python main.py
```

### Opciones del menú

| Opción | Descripción |
|--------|-------------|
| 1 | Resolver un problema personalizado (ingresar trabajos manualmente) |
| 2 | Ejecutar benchmark completo y generar gráficas PNG |
| 3 | Comparar calidad de soluciones DP vs LPT en instancias aleatorias |
| 4 | Ver diagrama de Gantt interactivo |
| 5 | Demo rápido con `n=10, m=2` |
| 6 | Análisis de escala logarítmica del DP |
| 7 | Demo de instancias adversariales (casos donde LPT es subóptimo) |
| 8 | Salir |

---

## Análisis de complejidad

| Algoritmo | Tiempo | Espacio | Óptimo |
|-----------|--------|---------|--------|
| Bitmask DP | `O(2^n)` | `O(2^n)` | Sí |
| LPT Greedy | `O(n log n)` | `O(n + m)` | No (aproximado) |

---

## Calidad de la solución Greedy

El algoritmo LPT garantiza teóricamente un ratio de aproximación de `4/3 - 1/(3m)`.  
Para `m=2` esto equivale a `≈ 1.167` en el peor caso.

En las pruebas empíricas realizadas, el ratio promedio medido fue inferior a `1.05`,
lo que significa que LPT se aleja menos del 5% del óptimo en la mayoría de los casos.