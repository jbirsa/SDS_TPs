#!/usr/bin/env python3
"""Regenera los gráficos del TP3 con múltiples corridas, barras de error,
criterio cuantitativo de estacionariedad y evolución temporal de <L>.

Produce los archivos con sufijo ``_FIXED`` en las mismas carpetas que los
originales (tp3-visual/1_input-t2, 2_input-t1, 3_input-k) y no toca ningún
gráfico existente.

Si no hay corridas en ``tp3-output-fixed/`` las genera invocando la
simulación Java (modalidades A y B, SERPENTINE, varias semillas).

Uso:
    python3 plot_sweep_fixed.py                # todo
    python3 plot_sweep_fixed.py t2             # sólo t2
    python3 plot_sweep_fixed.py --no-run       # no correr sims, sólo plotear
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

# ───────────────────────── Paths ─────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SDS_DIR   = REPO_ROOT / "tp3-sds"
OUT_ROOT  = REPO_ROOT / "tp3-output-fixed"

GRAPH_DIRS = {
    "t2": REPO_ROOT / "tp3-visual" / "1_input-t2",
    "t1": REPO_ROOT / "tp3-visual" / "2_input-t1",
    "k" : REPO_ROOT / "tp3-visual" / "3_input-k",
}

# ───────────────────── Simulation grid ─────────────────────
# mismos puntos que los gráficos originales
SWEEP_GRID = {
    "t2": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
           5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    "t1": [0.3, 0.6, 0.9, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
    "k" : [1, 2, 3, 4, 5, 6, 7, 8],
}
FIXED = {"t2": (1.0, None, 5),   # (t1, t2_variable, k)
         "t1": (None, 3.0, 5),
         "k" : (1.0, 3.0, None)}

N_SEEDS   = 30        # runs por punto
BASE_SEED = 1000
SIM_TIME  = 1000.0
MODALITIES = ("A", "B")

# ────────────────────── Plot style ─────────────────────────
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
MOD_SOLO_COLOR = {"A": "#1f77b4", "B": "#d62728"}
MOD_SOLO_MARKER = {"A": "o", "B": "o"}
MOD_SOLO_LINE  = {"A": "-", "B": "-"}

AVB_COLOR    = {"A": "#1f77b4", "B": "#d62728"}
AVB_MARKER   = {"A": "o", "B": "s"}
AVB_LINE     = {"A": "-", "B": "--"}

# Tamaños comunes para scatter + barras de error: marcadores chicos
# para que las barras (σ) queden bien visibles por encima del punto.
MARKER_SIZE = 4.0
ERR_CAPSIZE = 4.0
ERR_LW      = 1.3

# Paleta categórica para ⟨L⟩ vs tiempo: curvas claramente distinguibles.
CAT_PALETTE_10 = list(plt.get_cmap("tab10").colors)
CAT_PALETTE_20 = list(plt.get_cmap("tab20").colors)

def cat_color(i: int, n: int):
    pal = CAT_PALETTE_10 if n <= 10 else CAT_PALETTE_20
    return pal[i % len(pal)]

X_LABEL = {"t2": r"$t_2$ (s)",
           "t1": r"$t_1$ (s)",
           "k" : r"$k$ (servidores)"}

# ───────────────────── Stationarity ────────────────────────
# Criterio cuantitativo:
#   Ventana = último 50 % de la serie (t ∈ [T/2, T]).
#   Ajuste lineal L(t) = a + m·t en esa ventana → pendiente m (clientes/s).
#   La corrida se considera ESTACIONARIA iff
#           |m| < SLOPE_ABS                  (= 0.02 clientes/s)
#   y además el incremento lineal en la ventana, |m|·T_win, no supera
#   una fracción RANGE_FRAC del promedio μ en esa misma ventana cuando μ
#   es grande (protección contra rampas suaves sobre mean alto).
#   Esto distingue bien los regímenes ρ<1 (saturación, m≈0) de los ρ≥1
#   (crecimiento sostenido, m > 0.05 clientes/s típicamente).
STAT_WINDOW_FRAC = 0.5      # segunda mitad de la corrida
SLOPE_ABS        = 0.02     # clientes/s (umbral absoluto principal)
RANGE_FRAC       = 0.20     # tolera pendientes mayores solo si |m|·T_win < 20 % de μ

# ───────────────── simulation orchestration ────────────────

def sim_filename(mod: str, t1: float, t2: float, k: int) -> str:
    return f"out_{mod}_SERPENTINE_t1={t1:.2f}_t2={t2:.2f}_k={k}.txt"


def seed_dir(seed: int) -> Path:
    return OUT_ROOT / f"seed_{seed}"


def run_one(args):
    mod, t1, t2, k, seed = args
    outdir = seed_dir(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / sim_filename(mod, t1, t2, k)
    if outfile.exists() and outfile.stat().st_size > 1024:
        return str(outfile)
    env = os.environ.copy()
    cmd = [
        "java",
        "-Dframes.enabled=false",
        f"-Doutput.dir={outdir}",
        "-cp", "out", "simulation.Main",
        f"{t1:.2f}", f"{t2:.2f}", str(k), mod, "SERPENTINE",
        f"{SIM_TIME:.1f}", str(seed),
    ]
    subprocess.run(cmd, cwd=str(SDS_DIR), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=env)
    return str(outfile)


def all_sim_jobs():
    jobs = []
    for sweep, grid in SWEEP_GRID.items():
        t1f, t2f, kf = FIXED[sweep]
        for v in grid:
            if sweep == "t2": t1, t2, k = t1f, v, kf
            elif sweep == "t1": t1, t2, k = v, t2f, kf
            else:               t1, t2, k = t1f, t2f, int(v)
            for mod in MODALITIES:
                for s in range(N_SEEDS):
                    jobs.append((mod, t1, t2, k, BASE_SEED + s))
    # deduplicate
    return sorted(set(jobs))


def ensure_simulations():
    jobs = all_sim_jobs()
    todo = []
    for j in jobs:
        mod, t1, t2, k, seed = j
        outfile = seed_dir(seed) / sim_filename(mod, t1, t2, k)
        if not outfile.exists() or outfile.stat().st_size < 1024:
            todo.append(j)
    print(f"[sim] total jobs: {len(jobs)} | pendientes: {len(todo)}")
    if not todo:
        return
    workers = max(1, (os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, j) for j in todo]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  [sim] {done}/{len(todo)}")
    print("[sim] listo")

# ───────────────────── parsing ───────────────────────────

def parse_run(path: Path) -> dict:
    perm: list[float] = []
    ts: list[tuple[float, float]] = []
    stats: dict = {}
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
                        try:   stats[k] = float(v)
                        except ValueError: stats[k] = v
            elif section == "ts":
                p = line.split()
                if len(p) == 2:
                    ts.append((float(p[0]), float(p[1])))
            elif section == "perm":
                p = line.split()
                if len(p) == 3:
                    perm.append(float(p[2]))

    total_ql = float(sum(stats.get("qls", []) or [0.0]))
    times = np.array([p[0] for p in ts], dtype=float)
    lens  = np.array([p[1] for p in ts], dtype=float)
    return {
        "totalQL": total_ql,
        "avgPermanenceTime": float(stats.get("avgPermanenceTime", float("nan"))),
        "times": times,
        "lens":  lens,
        "permTimes": perm,
    }


# ───────────────────── analysis ──────────────────────────

def window_slope(times: np.ndarray, lens: np.ndarray,
                 frac: float = STAT_WINDOW_FRAC) -> tuple[float, float, float]:
    """Devuelve (slope, mean, t_window) ajustados sobre la cola de la serie."""
    if times.size < 4:
        return 0.0, 0.0, 0.0
    T = float(times[-1])
    cut = T * (1 - frac)
    mask = times >= cut
    if mask.sum() < 3:
        mask = np.ones_like(times, dtype=bool)
    t, l = times[mask], lens[mask]
    slope, _ = np.polyfit(t, l, 1)
    return float(slope), float(l.mean()), float(t[-1] - t[0])


def is_stationary(slope: float, mean: float, window: float) -> bool:
    if window <= 0:
        return False
    # Umbral principal: pendiente absoluta chica.
    if abs(slope) < SLOPE_ABS:
        return True
    # Escape para medias grandes: aún es estacionaria si la rampa en la
    # ventana representa una fracción pequeña del nivel medio.
    if mean > 5.0 and abs(slope) * window < RANGE_FRAC * mean:
        return True
    return False


# ───────────────── data aggregation ──────────────────────

def collect(sweep: str) -> dict:
    """Devuelve:
        data[mod] = list of dicts por valor de sweep:
            {x, slopes[], means[], perms[], stat_flags[], series:[(times,lens)]}
    """
    t1f, t2f, kf = FIXED[sweep]
    grid = SWEEP_GRID[sweep]
    data: dict[str, list[dict]] = {m: [] for m in MODALITIES}

    for v in grid:
        if sweep == "t2": t1, t2, k = t1f, v, kf
        elif sweep == "t1": t1, t2, k = v, t2f, kf
        else:               t1, t2, k = t1f, t2f, int(v)
        for mod in MODALITIES:
            slopes, means, perms, flags, total_qls, series = [], [], [], [], [], []
            for s in range(N_SEEDS):
                p = seed_dir(BASE_SEED + s) / sim_filename(mod, t1, t2, k)
                if not p.exists():
                    continue
                d = parse_run(p)
                slope, mu, win = window_slope(d["times"], d["lens"])
                slopes.append(slope)
                means.append(mu)
                total_qls.append(d["totalQL"])
                perms.append(d["avgPermanenceTime"])
                flags.append(is_stationary(slope, mu, win))
                series.append((d["times"], d["lens"]))
            data[mod].append({
                "x": v,
                "slopes":  np.array(slopes),
                "means":   np.array(means),
                "totalQL": np.array(total_qls),
                "perms":   np.array(perms),
                "flags":   np.array(flags, dtype=bool),
                "series":  series,
            })
    return data


def agg(arr: np.ndarray) -> tuple[float, float]:
    """Devuelve (media, desvío estándar σ) ignorando NaN.

    Convención única para todas las barras de error del TP: σ (no σ/√N).
    Refleja la variabilidad entre runs en lugar del error de la media.
    """
    if arr.size == 0:
        return float("nan"), 0.0
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return float("nan"), 0.0
    mu  = float(a.mean())
    sig = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return mu, sig


# ───────────────────── plotting ──────────────────────────

def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    fig.savefig(p, dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  -> {p.relative_to(REPO_ROOT)}")


def filename_prefix(sweep: str, kind: str, mod: str | None, avb: bool) -> str:
    """Replica el patrón usado en las carpetas.

    - En t2 los archivos _solo_ no llevan el sufijo _t2 en el nombre.
    - En t1/k sí lo llevan.
    """
    suffix = "" if sweep == "t2" else f"_{sweep}"
    if avb:
        return f"sweep_AvB_{sweep}_SERPENTINA_{kind}_FIXED.png"
    return f"sweep_{mod}{suffix}_SERPENTINA_{kind}_FIXED.png"


def ordered(name: str, n: int) -> str:
    """Antepone el número ordinal usado en los originales ('1_', '3_', ...)."""
    return f"{n}_{name}"


def _xticks(xs):
    return sorted({float(x) for x in xs})


def plot_L_vs_t(sweep: str, data: dict) -> None:
    """⟨L⟩ vs tiempo — una curva por valor del input (promedio entre runs).

    Para t2 se usan únicamente valores enteros (salto 1) para evitar curvas
    superpuestas. Para t1/k se usan todos los puntos del barrido.
    Los colores provienen de una paleta categórica (tab10/tab20).
    """
    if sweep == "t2":
        keep = lambda x: abs(x - round(x)) < 1e-9 and 1 <= int(round(x)) <= 8
    else:
        keep = lambda x: True

    for mod in MODALITIES:
        rows_all = data[mod]
        rows = [r for r in rows_all if keep(r["x"]) and r["series"]]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        n = len(rows)
        for i, r in enumerate(rows):
            times = r["series"][0][0]
            if times.size == 0:
                continue
            mat = np.full((len(r["series"]), times.size), np.nan)
            for j, (t, l) in enumerate(r["series"]):
                m = min(times.size, t.size)
                mat[j, :m] = l[:m]
            mean_L = np.nanmean(mat, axis=0)
            ax.plot(times, mean_L, color=cat_color(i, n),
                    linewidth=1.8,
                    label=f"{sweep}={r['x']:g}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Largo promedio de fila (clientes)")
        ax.legend(loc="upper left", ncol=2, fontsize=10,
                  title=X_LABEL[sweep])
        fig.tight_layout()
        _save(fig, GRAPH_DIRS[sweep],
              ordered(filename_prefix(sweep, "L_vs_tiempo", mod, False), 0))


def _empty_plot(sweep: str, ylabel: str, msg: str,
                outdir: Path, name: str) -> None:
    """Produce un plot con un mensaje cuando no hay datos válidos."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlabel(X_LABEL[sweep])
    ax.set_ylabel(ylabel)
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#555",
            wrap=True)
    fig.tight_layout()
    _save(fig, outdir, name)


def plot_L_mean(sweep: str, data: dict) -> None:
    """⟨L⟩ promedio vs input (sólo configuraciones estacionarias, con barras σ).

    Un punto (x, mod) se incluye sólo si la mayoría de sus corridas (≥50 %)
    satisfacen el criterio cuantitativo de estacionariedad. El promedio y la
    barra (σ) se computan únicamente sobre los runs estacionarios de ese punto.
    Si no hay ningún punto estacionario (modalidad B en ciertos barridos), se
    genera igualmente el archivo _FIXED con una anotación explicativa.
    """
    for mod in MODALITIES:
        rows = data[mod]
        xs, mus, errs = [], [], []
        for r in rows:
            stat = r["flags"]
            n = stat.size
            if n == 0 or stat.sum() / n < 0.5:
                continue
            mu, se = agg(r["means"][stat])
            xs.append(r["x"]); mus.append(mu); errs.append(se)
        prefix = {"A": 1, "B": 4}[mod]
        fname = ordered(filename_prefix(sweep, "largo_fila", mod, False), prefix)
        if not xs:
            _empty_plot(sweep, "Largo promedio de fila (clientes)",
                        f"Modalidad {mod}: sin configuraciones\n"
                        f"estacionarias en este barrido.\n"
                        "Ver gráfico de m vs input.",
                        GRAPH_DIRS[sweep], fname)
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.errorbar(xs, mus, yerr=errs, marker=MOD_SOLO_MARKER[mod],
                    linestyle=MOD_SOLO_LINE[mod], color=MOD_SOLO_COLOR[mod],
                    linewidth=2.0, markersize=MARKER_SIZE,
                    capsize=ERR_CAPSIZE, elinewidth=ERR_LW)
        ax.set_xlabel(X_LABEL[sweep])
        ax.set_ylabel("Largo promedio de fila (clientes)")
        ax.set_xticks(_xticks(xs))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        _save(fig, GRAPH_DIRS[sweep], fname)


def plot_m(sweep: str, data: dict) -> None:
    """Pendiente m vs input (TODOS los inputs)."""
    for mod in MODALITIES:
        rows = data[mod]
        xs, mus, errs = [], [], []
        for r in rows:
            if r["slopes"].size == 0:
                continue
            mu, se = agg(r["slopes"])
            xs.append(r["x"]); mus.append(mu); errs.append(se)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.axhline(0, color="#888", linestyle=":", linewidth=1.0)
        ax.errorbar(xs, mus, yerr=errs, marker=MOD_SOLO_MARKER[mod],
                    linestyle=MOD_SOLO_LINE[mod], color=MOD_SOLO_COLOR[mod],
                    linewidth=2.0, markersize=MARKER_SIZE,
                    capsize=ERR_CAPSIZE, elinewidth=ERR_LW)
        ax.set_xlabel(X_LABEL[sweep])
        ax.set_ylabel("Tasa de crecimiento del largo de fila (clientes/s)")
        ax.set_xticks(_xticks(xs))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        prefix = {"A": 1, "B": 4}[mod]
        _save(fig, GRAPH_DIRS[sweep],
              ordered(filename_prefix(sweep, "tasa_crecimiento", mod, False), prefix))


def plot_W(sweep: str, data: dict) -> None:
    """⟨W⟩ vs input (TODOS los inputs)."""
    for mod in MODALITIES:
        rows = data[mod]
        xs, mus, errs = [], [], []
        for r in rows:
            mu, se = agg(r["perms"])
            if not np.isfinite(mu):
                continue
            xs.append(r["x"]); mus.append(mu); errs.append(se)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.errorbar(xs, mus, yerr=errs, marker=MOD_SOLO_MARKER[mod],
                    linestyle=MOD_SOLO_LINE[mod], color=MOD_SOLO_COLOR[mod],
                    linewidth=2.0, markersize=MARKER_SIZE,
                    capsize=ERR_CAPSIZE, elinewidth=ERR_LW)
        ax.set_xlabel(X_LABEL[sweep])
        ax.set_ylabel("Tiempo de permanencia promedio (s)")
        ax.set_xticks(_xticks(xs))
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        prefix = {"A": 3, "B": 6}[mod]
        _save(fig, GRAPH_DIRS[sweep],
              ordered(filename_prefix(sweep, "permanencia", mod, False), prefix))


def plot_avb_L(sweep: str, data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    all_x = []
    for mod in MODALITIES:
        rows = data[mod]
        xs, mus, errs = [], [], []
        for r in rows:
            stat = r["flags"]
            n = stat.size
            if n == 0 or stat.sum() / n < 0.5:
                continue
            mu, se = agg(r["means"][stat])
            xs.append(r["x"]); mus.append(mu); errs.append(se)
        if not xs:
            continue
        ax.errorbar(xs, mus, yerr=errs, marker=AVB_MARKER[mod],
                    linestyle=AVB_LINE[mod], color=AVB_COLOR[mod],
                    linewidth=2.0, markersize=MARKER_SIZE,
                    capsize=ERR_CAPSIZE, elinewidth=ERR_LW,
                    label=f"Modalidad {mod}")
        all_x.extend(xs)
    ax.set_xlabel(X_LABEL[sweep])
    ax.set_ylabel("Largo promedio de fila (clientes)")
    if all_x:
        ax.set_xticks(_xticks(all_x))
        ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    _save(fig, GRAPH_DIRS[sweep],
          ordered(filename_prefix(sweep, "largo_fila", None, True), 7))


def _avb_plot(sweep: str, data: dict, *,
              per_row, ylabel: str, fname: str,
              zero_line: bool = False) -> None:
    """Dibuja A y B en el mismo eje usando los mismos datos ya calculados.

    ``per_row(row) -> (mu, err) | None`` decide qué agregado usar para cada
    punto y permite filtrar (devolviendo None).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if zero_line:
        ax.axhline(0, color="#888", linestyle=":", linewidth=1.0)
    all_x: list[float] = []
    for mod in MODALITIES:
        xs, mus, errs = [], [], []
        for r in data[mod]:
            agg_pair = per_row(r)
            if agg_pair is None:
                continue
            mu, se = agg_pair
            if not np.isfinite(mu):
                continue
            xs.append(r["x"]); mus.append(mu); errs.append(se)
        if not xs:
            continue
        ax.errorbar(xs, mus, yerr=errs, marker=AVB_MARKER[mod],
                    linestyle=AVB_LINE[mod], color=AVB_COLOR[mod],
                    linewidth=2.0, markersize=MARKER_SIZE,
                    capsize=ERR_CAPSIZE, elinewidth=ERR_LW,
                    label=f"Modalidad {mod}")
        all_x.extend(xs)
    ax.set_xlabel(X_LABEL[sweep])
    ax.set_ylabel(ylabel)
    if all_x:
        ax.set_xticks(_xticks(all_x))
        ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    _save(fig, GRAPH_DIRS[sweep], fname)


def plot_avb_L_vs_input(sweep: str, data: dict) -> None:
    """Comparativo A vs B de ⟨L⟩ vs input (solo puntos estacionarios)."""
    def per(row):
        stat = row["flags"]
        n = stat.size
        if n == 0 or stat.sum() / n < 0.5:
            return None
        return agg(row["means"][stat])
    _avb_plot(sweep, data, per_row=per,
              ylabel="Largo promedio de fila (clientes)",
              fname=f"L_vs_{sweep}_FIXED_AVB.png")


def plot_avb_W_vs_input(sweep: str, data: dict) -> None:
    """Comparativo A vs B de ⟨W⟩ vs input (todos los puntos)."""
    def per(row):
        if row["perms"].size == 0:
            return None
        return agg(row["perms"])
    _avb_plot(sweep, data, per_row=per,
              ylabel="Tiempo de permanencia promedio (s)",
              fname=f"W_vs_{sweep}_FIXED_AVB.png")


def plot_avb_m_vs_input(sweep: str, data: dict) -> None:
    """Comparativo A vs B de pendiente m vs input (todos los puntos)."""
    def per(row):
        if row["slopes"].size == 0:
            return None
        return agg(row["slopes"])
    _avb_plot(sweep, data, per_row=per,
              ylabel="Tasa de crecimiento del largo de fila (clientes/s)",
              fname=f"m_vs_{sweep}_FIXED_AVB.png",
              zero_line=True)


# ────────────────────── main ────────────────────────────

def main() -> None:
    sweeps = ("t2", "t1", "k")
    no_run = False
    args = list(sys.argv[1:])
    if "--no-run" in args:
        no_run = True
        args.remove("--no-run")
    if args:
        sweeps = tuple(a for a in args if a in ("t2", "t1", "k"))

    if not no_run:
        ensure_simulations()

    summary: dict = {}
    for sweep in sweeps:
        print(f"\n=== {sweep} ===")
        data = collect(sweep)
        summary[sweep] = data
        plot_L_vs_t(sweep, data)
        plot_L_mean(sweep, data)
        plot_m(sweep, data)
        plot_W(sweep, data)
        plot_avb_L(sweep, data)
        plot_avb_L_vs_input(sweep, data)
        plot_avb_W_vs_input(sweep, data)
        plot_avb_m_vs_input(sweep, data)

    # resumen de estacionariedad
    print("\n========== Estacionariedad ==========")
    for sweep, data in summary.items():
        for mod in MODALITIES:
            for r in data[mod]:
                n = r["flags"].size
                if n == 0:
                    continue
                frac = r["flags"].sum() / n
                label = "ESTACIONARIA" if frac >= 0.5 else "NO ESTACIONARIA"
                print(f"  {sweep}={r['x']:g} mod={mod}: "
                      f"{r['flags'].sum()}/{n} runs estacionarios → {label}")

    print("\nListo.")


if __name__ == "__main__":
    main()
