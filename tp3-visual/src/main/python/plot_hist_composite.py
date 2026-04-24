#!/usr/bin/env python3
"""Genera figuras compuestas de histogramas de tiempos de permanencia.

Para cada combinación (sweep, modalidad) produce una única figura con
3 subplots (layout 2+1) correspondientes a los 3 valores del parámetro
que ya se mostraban en la presentación.

Agrega los tiempos de permanencia de las 30 seeds en tp3-output-fixed/.

Salida: tp3-visual/graphs/histogramas_{sweep}_{mod}_FIXED.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUT_ROOT = REPO_ROOT / "tp3-output-fixed"
GRAPHS_DIR = REPO_ROOT / "tp3-visual" / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

MOD_COLOR = {"A": "#1f77b4", "B": "#d62728"}

TARGETS = {
    "t2": [1, 4, 8],
    "t1": [0.3, 1.8, 3.0],
    "k":  [1, 4, 8],
}

LABEL_FMT = {
    "t2": r"$t_2 = {v:g}\,$s",
    "t1": r"$t_1 = {v:g}\,$s",
    "k":  r"$k = {v:g}$",
}

N_BINS = 40


def sim_filename(mod: str, t1: float, t2: float, k: int) -> str:
    return f"out_{mod}_SERPENTINE_t1={t1:.2f}_t2={t2:.2f}_k={k}.txt"


def params_for(sweep: str, v: float) -> tuple[float, float, int]:
    """Devuelve (t1, t2, k) para el caso variando `sweep` a `v`."""
    if sweep == "t2":
        return 1.0, float(v), 5
    if sweep == "t1":
        return float(v), 3.0, 5
    if sweep == "k":
        return 1.0, 3.0, int(v)
    raise ValueError(sweep)


def read_permanence(path: Path) -> list[float]:
    perm: list[float] = []
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
            if section == "perm":
                parts = line.split()
                if len(parts) == 3:
                    try:
                        perm.append(float(parts[2]))
                    except ValueError:
                        pass
    return perm


def load_aggregated(mod: str, sweep: str, v: float) -> np.ndarray:
    t1, t2, k = params_for(sweep, v)
    fname = sim_filename(mod, t1, t2, k)
    all_perm: list[float] = []
    seed_dirs = sorted(OUT_ROOT.glob("seed_*"))
    for sdir in seed_dirs:
        f = sdir / fname
        if f.exists():
            all_perm.extend(read_permanence(f))
    return np.array(all_perm, dtype=float)


def make_composite(sweep: str, mod: str) -> Path:
    color = MOD_COLOR[mod]
    vals = TARGETS[sweep]
    data = [load_aggregated(mod, sweep, v) for v in vals]
    sizes = [d.size for d in data]
    print(f"  {sweep}/{mod}: sizes = {sizes}")

    concat = np.concatenate([d for d in data if d.size > 0])
    if concat.size == 0:
        raise RuntimeError(f"sin datos para {sweep}/{mod}")

    # Rango común de X basado en percentil 99 global (robusto a outliers).
    x_min = 0.0
    x_max = float(np.percentile(concat, 99.0))
    if x_max <= 0:
        x_max = float(concat.max()) if concat.size else 1.0
    bins = np.linspace(x_min, x_max, N_BINS + 1)

    # Layout 2+1 via GridSpec: 2 arriba + 1 abajo centrado.
    # Aspecto ~1.75 (más ancho que alto) para que quede cómodo en slide 4:3.
    fig = plt.figure(figsize=(14.0, 8.0))
    gs = fig.add_gridspec(2, 4, wspace=0.22, hspace=0.32,
                          left=0.055, right=0.985, top=0.985, bottom=0.09)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax2 = fig.add_subplot(gs[1, 1:3])
    axes = [ax0, ax1, ax2]

    for ax, v, d in zip(axes, vals, data):
        if d.size > 0:
            ax.hist(d, bins=bins, color=color, edgecolor="white",
                    linewidth=0.4, alpha=0.9)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
        label = LABEL_FMT[sweep].format(v=v)
        ax.text(0.97, 0.93, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.32", fc="white",
                          ec="#888", alpha=0.85, lw=0.6))
        ax.set_xlabel("Tiempo de permanencia (s)", fontsize=13)
        ax.set_ylabel("Frecuencia", fontsize=13)

    out = GRAPHS_DIR / f"histogramas_{sweep}_{mod}_FIXED.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out.name}  (xmax={x_max:.1f} s, bins={N_BINS}, layout=2+1)")
    return out


def main() -> None:
    for sweep in ("t2", "t1", "k"):
        for mod in ("A", "B"):
            make_composite(sweep, mod)


if __name__ == "__main__":
    main()
