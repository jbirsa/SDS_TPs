#!/usr/bin/env python3
"""Genera los graficos del barrido en t1 o t2 para una modalidad dada (A o B).

Uso:
  python3 plot_sweep.py <A|B> <t1|t2>

Configuraciones:
  - sweep t2 : t1=1, k=5, t2 variando
  - sweep t1 : t2=3, k=5, t1 variando

Para cada tipo de fila (FREE y SERPENTINA) produce:
  * largo promedio de fila vs <sweep>
  * tasa de crecimiento (pendiente del largo de fila vs tiempo) vs <sweep>
  * histograma de tiempos de permanencia para varios valores del parametro
  * tiempo de permanencia promedio vs <sweep>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_DIR = REPO_ROOT / "tp3-output"
GRAPHS_DIR = REPO_ROOT / "tp3-visual" / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

QT_COLORS = {"FREE": "#1f77b4", "SERPENTINA": "#d62728"}

MODALITY = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
if MODALITY not in ("A", "B"):
    sys.exit("modalidad invalida: usar 'A' o 'B'")

SWEEP = sys.argv[2].lower() if len(sys.argv) > 2 else "t2"
if SWEEP not in ("t1", "t2", "k"):
    sys.exit("parametro invalido: usar 't1', 't2' o 'k'")

if SWEEP == "t2":
    HIST_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    X_LABEL = r"$t_2$ (s)"
    FILE_PAT = re.compile(
        rf"out_{MODALITY}_(?P<qt>FREE|SERPENTINA)_t1=1\.00_t2=(?P<val>[\d.]+)_k=5\.txt"
    )
    GLOB = f"out_{MODALITY}_*_t1=1.00_t2=*_k=5.txt"
elif SWEEP == "t1":
    HIST_VALUES = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    X_LABEL = r"$t_1$ (s)"
    FILE_PAT = re.compile(
        rf"out_{MODALITY}_(?P<qt>FREE|SERPENTINA)_t1=(?P<val>[\d.]+)_t2=3\.00_k=5\.txt"
    )
    GLOB = f"out_{MODALITY}_*_t1=*_t2=3.00_k=5.txt"
else:  # k
    HIST_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
    X_LABEL = r"$k$ (servidores)"
    FILE_PAT = re.compile(
        rf"out_{MODALITY}_(?P<qt>FREE|SERPENTINA)_t1=1\.00_t2=3\.00_k=(?P<val>\d+)\.txt"
    )
    GLOB = f"out_{MODALITY}_*_t1=1.00_t2=3.00_k=*.txt"


def parse_run(path: Path) -> dict:
    perm_times: list[float] = []
    qt_series: list[tuple[float, float]] = []
    stats: dict[str, float] = {}

    section = None
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "STATS":
                section = "stats"
                continue
            if line == "QUEUE_TIMESERIES":
                section = "ts"
                continue
            if line == "PERMANENCE_TIMES":
                section = "perm"
                continue

            if section == "stats":
                for tok in line.split():
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    if k.startswith("avgQueueLength"):
                        stats.setdefault("qls", []).append(float(v))
                    else:
                        try:
                            stats[k] = float(v)
                        except ValueError:
                            stats[k] = v
            elif section == "ts":
                parts = line.split()
                if len(parts) == 2:
                    qt_series.append((float(parts[0]), float(parts[1])))
            elif section == "perm":
                parts = line.split()
                if len(parts) == 3:
                    perm_times.append(float(parts[2]))

    total_ql = float(sum(stats.get("qls", []) or [0.0]))

    growth_rate = 0.0
    if len(qt_series) >= 4:
        times = np.array([p[0] for p in qt_series], dtype=float)
        lens = np.array([p[1] for p in qt_series], dtype=float)
        half = len(times) // 2
        slope, _ = np.polyfit(times[half:], lens[half:], 1)
        growth_rate = float(slope)

    return {
        "totalQL": total_ql,
        "avgPermanenceTime": float(stats.get("avgPermanenceTime", float("nan"))),
        "growthRate": growth_rate,
        "permTimes": perm_times,
        "qtSeries": qt_series,
    }


def load_sweep() -> dict[str, list[dict]]:
    by_qt: dict[str, list[dict]] = {"FREE": [], "SERPENTINA": []}
    for path in sorted(OUTPUT_DIR.glob(GLOB)):
        m = FILE_PAT.match(path.name)
        if not m:
            continue
        qt = m.group("qt")
        val = float(m.group("val"))
        row = {"x": val, "path": path}
        row.update(parse_run(path))
        by_qt[qt].append(row)
    for qt in by_qt:
        by_qt[qt].sort(key=lambda r: r["x"])
    return by_qt


def _save(fig, name: str) -> None:
    out = GRAPHS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


def plot_queue_length(rows: list[dict], qt: str) -> None:
    xs = [r["x"] for r in rows]
    ys = [r["totalQL"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7, color=QT_COLORS[qt])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Largo promedio de fila (clientes)")
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, f"sweep_{MODALITY}_{SWEEP}_{qt}_largo_fila.png")


def plot_growth_rate(rows: list[dict], qt: str) -> None:
    xs = [r["x"] for r in rows]
    ys = [r["growthRate"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(0, color="#888", linestyle=":", linewidth=1.0)
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7, color=QT_COLORS[qt])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Tasa de crecimiento del largo de fila (clientes/s)")
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, f"sweep_{MODALITY}_{SWEEP}_{qt}_tasa_crecimiento.png")


def plot_permanence_mean(rows: list[dict], qt: str) -> None:
    xs = [r["x"] for r in rows]
    ys = [r["avgPermanenceTime"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7, color=QT_COLORS[qt])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Tiempo de permanencia promedio (s)")
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, f"sweep_{MODALITY}_{SWEEP}_{qt}_permanencia.png")


def plot_permanence_histogram(rows: list[dict], qt: str, target: float) -> None:
    row = min(rows, key=lambda r: abs(r["x"] - target))
    perm = row.get("permTimes") or []
    if not perm:
        print(f"  [{qt}] {SWEEP}={target}: sin datos de permanencia")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(perm, bins=30, color=QT_COLORS[qt], edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Tiempo de permanencia (s)")
    ax.set_ylabel("Frecuencia (clientes)")
    fig.tight_layout()
    _save(fig, f"sweep_{MODALITY}_{SWEEP}_{qt}_hist_permanencia_{SWEEP}={row['x']:g}.png")


def main() -> None:
    print(f"Cargando desde: {OUTPUT_DIR} | modalidad={MODALITY} sweep={SWEEP}")
    data = load_sweep()
    for qt in ("FREE", "SERPENTINA"):
        rows = data.get(qt, [])
        if not rows:
            print(f"  [{qt}] sin datos")
            continue
        print(f"\n{qt}: {len(rows)} runs")
        plot_queue_length(rows, qt)
        plot_growth_rate(rows, qt)
        plot_permanence_mean(rows, qt)
        for v in HIST_VALUES:
            plot_permanence_histogram(rows, qt, v)
    print("\nListo.")


if __name__ == "__main__":
    main()
