#!/usr/bin/env python3
"""Heatmaps del TP3: k=5 fijo, (t1, t2) variable simultáneamente.

Genera 6 mapas de calor (modalidades A y B, SERPENTINE):

  - heatmap_L_{A,B}_FIXED.png : ⟨L⟩ promedio (solo celdas estacionarias).
                                 Celdas no estacionarias → NaN en gris claro.
  - heatmap_m_{A,B}_FIXED.png : tasa de crecimiento m (todas las celdas,
                                 cmap divergente centrado en 0).
  - heatmap_W_{A,B}_FIXED.png : tiempo de permanencia promedio ⟨W⟩
                                 (todas las celdas).

A y B comparten la misma escala de color por métrica (vmin/vmax comunes),
para permitir comparación visual directa entre modalidades.

Si faltan corridas en tp3-output-fixed/ las genera invocando la simulación Java.

Uso:
    python3 plot_heatmaps.py           # corre sims faltantes y plotea
    python3 plot_heatmaps.py --no-run  # solo plotea con lo que haya
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reutilizamos parsing, slope y criterio de estacionariedad.
from plot_sweep_fixed import (
    parse_run, window_slope, is_stationary,
    OUT_ROOT, SDS_DIR, REPO_ROOT, sim_filename, seed_dir,
)

# ─────────────────────── Grilla de la heatmap ───────────────────────
# Mismos rangos que el resto del TP (t1∈[0.3,3.0], t2∈[1.0,8.0]).
T1_VALUES  = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
T2_VALUES  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
K_FIXED    = 5
MODALITIES = ("A", "B")
QTYPE      = "SERPENTINE"

N_SEEDS   = 30
BASE_SEED = 1000
SIM_TIME  = 1000.0

OUT_DIR = REPO_ROOT / "tp3-visual" / "graphs" / "heatmaps"

# ─────────────────── estilo paper ───────────────────
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": False,
})


# ─────────────────── simulación ───────────────────
def run_one(args):
    mod, t1, t2, seed = args
    outdir = seed_dir(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / sim_filename(mod, t1, t2, K_FIXED)
    if outfile.exists() and outfile.stat().st_size > 1024:
        return str(outfile)
    cmd = [
        "java",
        "-Dframes.enabled=false",
        f"-Doutput.dir={outdir}",
        "-cp", "out", "simulation.Main",
        f"{t1:.2f}", f"{t2:.2f}", str(K_FIXED), mod, QTYPE,
        f"{SIM_TIME:.1f}", str(seed),
    ]
    subprocess.run(cmd, cwd=str(SDS_DIR), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(outfile)


def ensure_simulations() -> int:
    """Corre las simulaciones faltantes. Devuelve cuántas se ejecutaron."""
    jobs = []
    for mod in MODALITIES:
        for t1 in T1_VALUES:
            for t2 in T2_VALUES:
                for s in range(N_SEEDS):
                    seed = BASE_SEED + s
                    p = seed_dir(seed) / sim_filename(mod, t1, t2, K_FIXED)
                    if not p.exists() or p.stat().st_size < 1024:
                        jobs.append((mod, t1, t2, seed))
    total = len(T1_VALUES) * len(T2_VALUES) * N_SEEDS * len(MODALITIES)
    print(f"[sim] grid: {len(MODALITIES)} mods × {len(T1_VALUES)}×{len(T2_VALUES)} "
          f"× {N_SEEDS} seeds = {total} → pendientes: {len(jobs)}")
    if not jobs:
        return 0
    workers = max(1, (os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  [sim] {done}/{len(jobs)}")
    print("[sim] listo")
    return len(jobs)


# ─────────────────── agregación por celda ───────────────────
def cell_aggregates(mod: str, t1: float, t2: float):
    """Devuelve (L_mean_if_stationary_or_nan, m_mean, W_mean)."""
    Ls, ms, Ws = [], [], []
    stat_flags = []
    for s in range(N_SEEDS):
        p = seed_dir(BASE_SEED + s) / sim_filename(mod, t1, t2, K_FIXED)
        if not p.exists():
            continue
        d = parse_run(p)
        slope, mu, win = window_slope(d["times"], d["lens"])
        Ls.append(mu)
        ms.append(slope)
        Ws.append(d["avgPermanenceTime"])
        stat_flags.append(is_stationary(slope, mu, win))

    if not Ls:
        return float("nan"), float("nan"), float("nan")

    Ls = np.array(Ls)
    ms = np.array(ms)
    Ws = np.array(Ws, dtype=float)
    stat_flags = np.array(stat_flags, dtype=bool)

    # ⟨L⟩: solo si la mayoría de runs son estacionarias; promedio sobre estacionarias.
    if stat_flags.sum() / stat_flags.size >= 0.5:
        L_stat = Ls[stat_flags]
        L_val = float(np.nanmean(L_stat)) if L_stat.size else float("nan")
    else:
        L_val = float("nan")

    m_val = float(np.nanmean(ms))
    W_val = float(np.nanmean(Ws[np.isfinite(Ws)])) if np.any(np.isfinite(Ws)) else float("nan")
    return L_val, m_val, W_val


def build_matrices(mod: str):
    nT1, nT2 = len(T1_VALUES), len(T2_VALUES)
    # Filas = t2 (eje Y), columnas = t1 (eje X).
    L_mat = np.full((nT2, nT1), np.nan)
    m_mat = np.full((nT2, nT1), np.nan)
    W_mat = np.full((nT2, nT1), np.nan)
    for i, t1 in enumerate(T1_VALUES):
        for j, t2 in enumerate(T2_VALUES):
            L, m, W = cell_aggregates(mod, t1, t2)
            L_mat[j, i] = L
            m_mat[j, i] = m
            W_mat[j, i] = W
    return L_mat, m_mat, W_mat


# ─────────────────── plotting ───────────────────
def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  -> {p.relative_to(REPO_ROOT)}")


def _edges(vals):
    """Bordes de celda para pcolormesh con celdas centradas en vals."""
    v = np.asarray(vals, dtype=float)
    mids = 0.5 * (v[1:] + v[:-1])
    first = v[0] - (mids[0] - v[0])
    last  = v[-1] + (v[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_heatmap(mat: np.ndarray, *, cmap, cbar_label: str, fname: str,
                 vmin: float, vmax: float) -> None:
    xedges = _edges(T1_VALUES)
    yedges = _edges(T2_VALUES)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")  # NaN: gris claro, distinto del colormap
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    mesh = ax.pcolormesh(xedges, yedges, masked, cmap=cmap_obj,
                         vmin=vmin, vmax=vmax, shading="flat")
    ax.set_xlabel(r"$t_1$ (s)")
    ax.set_ylabel(r"$t_2$ (s)")
    ax.set_xticks(T1_VALUES)
    ax.set_yticks(T2_VALUES)
    ax.tick_params(axis="x", rotation=45)
    ax.set_aspect("auto")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    _save(fig, fname)


def _shared_range(mats) -> tuple[float, float]:
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in mats])
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())


def _shared_symmetric(mats) -> tuple[float, float]:
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in mats])
    if vals.size == 0:
        return -1.0, 1.0
    vmax = float(np.max(np.abs(vals)))
    if vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def main():
    no_run = "--no-run" in sys.argv

    n_new = 0
    if not no_run:
        n_new = ensure_simulations()

    print("\n[agg] construyendo matrices...")
    mats = {}
    for mod in MODALITIES:
        L, m, W = build_matrices(mod)
        mats[mod] = {"L": L, "m": m, "W": W}
        n_stat = int(np.isfinite(L).sum())
        print(f"  mod={mod}: celdas estacionarias = {n_stat}/{L.size}")

    # Escalas compartidas A↔B por métrica.
    L_vmin, L_vmax = _shared_range([mats["A"]["L"], mats["B"]["L"]])
    W_vmin, W_vmax = _shared_range([mats["A"]["W"], mats["B"]["W"]])
    m_vmin, m_vmax = _shared_symmetric([mats["A"]["m"], mats["B"]["m"]])

    print("\n[scale] escalas compartidas:")
    print(f"  ⟨L⟩:  vmin={L_vmin:.4f}  vmax={L_vmax:.4f}")
    print(f"  m  :  vmin={m_vmin:.4f}  vmax={m_vmax:.4f} (centrado en 0)")
    print(f"  ⟨W⟩:  vmin={W_vmin:.4f}  vmax={W_vmax:.4f}")

    for mod in MODALITIES:
        plot_heatmap(mats[mod]["L"], cmap="viridis",
                     cbar_label=r"$\langle L \rangle$ (clientes)",
                     fname=f"heatmap_L_{mod}_FIXED.png",
                     vmin=L_vmin, vmax=L_vmax)
        plot_heatmap(mats[mod]["m"], cmap="coolwarm",
                     cbar_label=r"$m$ (clientes/s)",
                     fname=f"heatmap_m_{mod}_FIXED.png",
                     vmin=m_vmin, vmax=m_vmax)
        plot_heatmap(mats[mod]["W"], cmap="plasma",
                     cbar_label=r"$\langle W \rangle$ (s)",
                     fname=f"heatmap_W_{mod}_FIXED.png",
                     vmin=W_vmin, vmax=W_vmax)

    # ─────────── Resumen final ───────────
    print("\n========== RESUMEN ==========")
    print(f"Simulaciones nuevas ejecutadas: {n_new}")
    for mod in MODALITIES:
        n_stat = int(np.isfinite(mats[mod]["L"]).sum())
        print(f"Celdas estacionarias mod {mod}: {n_stat}/{mats[mod]['L'].size}")
    print("Escalas compartidas A↔B:")
    print(f"  ⟨L⟩: [{L_vmin:.4f}, {L_vmax:.4f}]  (cmap viridis)")
    print(f"  m  : [{m_vmin:.4f}, {m_vmax:.4f}]  (cmap coolwarm, centrado en 0)")
    print(f"  ⟨W⟩: [{W_vmin:.4f}, {W_vmax:.4f}]  (cmap plasma)")
    print("Confirmado: A y B comparten exactamente las mismas escalas por métrica.")


if __name__ == "__main__":
    main()
