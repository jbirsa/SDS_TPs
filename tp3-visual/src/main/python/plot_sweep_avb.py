#!/usr/bin/env python3
"""Superpone modalidad A y B en los graficos del barrido.

Uso:
  python3 plot_sweep_avb.py <t1|t2|k>

Genera por cada tipo de fila (FREE y SERPENTINA):
  * largo promedio de fila vs <sweep> — A y B en un mismo eje
  * tasa de crecimiento vs <sweep>
  * permanencia promedio vs <sweep>
  * histograma de permanencia para cada valor de HIST_VALUES (A y B superpuestos)
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

MOD_COLORS = {"A": "#1f77b4", "B": "#d62728"}
MOD_MARKERS = {"A": "o", "B": "s"}
MOD_LINESTYLES = {"A": "-", "B": "--"}

SWEEP = sys.argv[1].lower() if len(sys.argv) > 1 else "t2"
if SWEEP not in ("t1", "t2", "k"):
    sys.exit("parametro invalido: usar 't1', 't2' o 'k'")


def sweep_config(modality: str):
    if SWEEP == "t2":
        hist_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x_label = r"$t_2$ (s)"
        file_pat = re.compile(
            rf"out_{modality}_(?P<qt>FREE|SERPENTINA)_t1=1\.00_t2=(?P<val>[\d.]+)_k=5\.txt"
        )
        glob = f"out_{modality}_*_t1=1.00_t2=*_k=5.txt"
    elif SWEEP == "t1":
        hist_values = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
        x_label = r"$t_1$ (s)"
        file_pat = re.compile(
            rf"out_{modality}_(?P<qt>FREE|SERPENTINA)_t1=(?P<val>[\d.]+)_t2=3\.00_k=5\.txt"
        )
        glob = f"out_{modality}_*_t1=*_t2=3.00_k=5.txt"
    else:
        hist_values = [1, 2, 3, 4, 5, 6, 7, 8]
        x_label = r"$k$ (servidores)"
        file_pat = re.compile(
            rf"out_{modality}_(?P<qt>FREE|SERPENTINA)_t1=1\.00_t2=3\.00_k=(?P<val>\d+)\.txt"
        )
        glob = f"out_{modality}_*_t1=1.00_t2=3.00_k=*.txt"
    return hist_values, x_label, file_pat, glob


HIST_VALUES, X_LABEL, _, _ = sweep_config("A")


def parse_run(path: Path) -> dict:
    perm_times: list[float] = []
    qt_series: list[tuple[float, float]] = []
    stats: dict[str, float] = {}

    section = None
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "STATS":
                section = "stats"; continue
            if line == "QUEUE_TIMESERIES":
                section = "ts"; continue
            if line == "PERMANENCE_TIMES":
                section = "perm"; continue

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
    }


def load_for(modality: str) -> dict[str, list[dict]]:
    _, _, pat, glob = sweep_config(modality)
    by_qt: dict[str, list[dict]] = {"FREE": [], "SERPENTINA": []}
    for path in sorted(OUTPUT_DIR.glob(glob)):
        m = pat.match(path.name)
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


def _xticks(xs: list[float]):
    uniq = sorted(set(xs))
    return uniq


def plot_line(data_ab: dict[str, list[dict]], qt: str, key: str,
              ylabel: str, fname_suffix: str, zero_line: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if zero_line:
        ax.axhline(0, color="#888", linestyle=":", linewidth=1.0)
    all_x: list[float] = []
    for mod in ("A", "B"):
        rows = data_ab[mod].get(qt, [])
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, marker=MOD_MARKERS[mod], linestyle=MOD_LINESTYLES[mod],
                linewidth=2.0, markersize=7, color=MOD_COLORS[mod],
                label=f"Modalidad {mod}")
        all_x.extend(xs)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(ylabel)
    if all_x:
        ax.set_xticks(_xticks(all_x))
        ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    _save(fig, f"sweep_AvB_{SWEEP}_{qt}_{fname_suffix}.png")


def plot_hist(data_ab: dict[str, list[dict]], qt: str, target: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    all_vals: list[float] = []
    selected: dict[str, dict] = {}
    for mod in ("A", "B"):
        rows = data_ab[mod].get(qt, [])
        if not rows:
            continue
        row = min(rows, key=lambda r: abs(r["x"] - target))
        if row.get("permTimes"):
            selected[mod] = row
            all_vals.extend(row["permTimes"])

    if not all_vals:
        plt.close(fig)
        print(f"  [{qt}] {SWEEP}={target}: sin datos de permanencia")
        return

    bins = np.linspace(min(all_vals), max(all_vals), 30)
    for mod, row in selected.items():
        ax.hist(row["permTimes"], bins=bins,
                color=MOD_COLORS[mod], alpha=0.55,
                edgecolor="white", linewidth=0.4,
                label=f"Modalidad {mod}")
    ax.set_xlabel("Tiempo de permanencia (s)")
    ax.set_ylabel("Frecuencia (clientes)")
    ax.legend()
    fig.tight_layout()
    x_val = next(iter(selected.values()))["x"]
    _save(fig, f"sweep_AvB_{SWEEP}_{qt}_hist_permanencia_{SWEEP}={x_val:g}.png")


def main() -> None:
    print(f"Cargando desde: {OUTPUT_DIR} | sweep={SWEEP}")
    data_ab = {"A": load_for("A"), "B": load_for("B")}
    for qt in ("FREE", "SERPENTINA"):
        has_any = any(data_ab[m].get(qt) for m in ("A", "B"))
        if not has_any:
            print(f"  [{qt}] sin datos")
            continue
        print(f"\n{qt}")
        plot_line(data_ab, qt, "totalQL",
                  "Largo promedio de fila (clientes)", "largo_fila")
        plot_line(data_ab, qt, "growthRate",
                  "Tasa de crecimiento del largo de fila (clientes/s)",
                  "tasa_crecimiento", zero_line=True)
        plot_line(data_ab, qt, "avgPermanenceTime",
                  "Tiempo de permanencia promedio (s)", "permanencia")
        for v in HIST_VALUES:
            plot_hist(data_ab, qt, v)
    print("\nListo.")


if __name__ == "__main__":
    main()
