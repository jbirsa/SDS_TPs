#!/usr/bin/env python3
"""
Post-processing TP3 (Sistema 2 — Filas inteligentes).

Produce un gráfico por observable y por estudio (2.1 / 2.2 / 2.3), comparando
Modalidad A (una fila por servidor) y Modalidad B (fila compartida).

Observables por estudio:
  * Largo promedio de fila (régimen estacionario)
  * Tasa de crecimiento de la fila (régimen inestable)
  * Tiempo de permanencia promedio
  * Histograma de tiempos de permanencia (ejemplo representativo, estable)
  * Histograma de tiempos de permanencia (ejemplo representativo, inestable)

Los estados "estable" / "inestable" se detectan ajustando una regresión lineal
a la segunda mitad de la serie temporal de largo de fila (QUEUE_TIMESERIES).

Archivos generados en tp3-visual/graphs/:
  2_1_largo_fila.png          — largo promedio vs t2   (solo runs estables)
  2_1_tasa_crecimiento.png    — tasa clientes/s vs t2  (solo runs inestables)
  2_1_permanencia.png         — permanencia promedio vs t2
  2_1_hist_permanencia_estable.png
  2_1_hist_permanencia_inestable.png
  (análogamente 2_2_*  y  2_3_*)
"""

import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TP3_VISUAL = SCRIPT_DIR.parents[2]
GRAPHS_DIR = TP3_VISUAL / 'graphs'
REPO_ROOT  = TP3_VISUAL.parent
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / 'tp3-output'

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot styling (no titles, simple legends, readable fonts) ────────────────
plt.rcParams.update({
    'font.size':        13,
    'axes.labelsize':   14,
    'axes.titlesize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'axes.grid':        True,
    'grid.alpha':       0.3,
})

# ── Stability detection ─────────────────────────────────────────────────────
UNSTABLE_SLOPE_THRESHOLD = 0.05  # clientes/s

_FILE_PAT = re.compile(
    r'out_(?P<modality>[AB])_(?P<queueType>[A-Z]+)'
    r'_t1=(?P<t1>[\d.]+)_t2=(?P<t2>[\d.]+)_k=(?P<k>\d+)\.txt'
)

# ── Parser ──────────────────────────────────────────────────────────────────

def _parse_stats(path):
    data = {}
    perm = []
    qt_series = []

    size = Path(path).stat().st_size
    tail_bytes = min(size, 1024 * 1024)
    with open(path, 'rb') as fh:
        fh.seek(size - tail_bytes)
        raw_tail = fh.read().decode('utf-8', errors='replace')

    stats_idx = raw_tail.rfind('\nSTATS\n')
    if stats_idx < 0:
        stats_idx = raw_tail.find('STATS\n')
        if stats_idx < 0:
            return data
    tail = raw_tail[stats_idx:].lstrip('\n')

    in_perm = False
    in_ts   = False
    for raw in tail.splitlines():
        line = raw.rstrip()
        if line == 'STATS':
            in_perm = False; in_ts = False
            continue
        if line == 'PERMANENCE_TIMES':
            in_perm = True; in_ts = False
            continue
        if line == 'QUEUE_TIMESERIES':
            in_perm = False; in_ts = True
            continue

        if in_perm:
            parts = line.split()
            if len(parts) == 3:
                perm.append(float(parts[2]))
            continue

        if in_ts:
            parts = line.split()
            if len(parts) == 2:
                try:
                    qt_series.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            continue

        for tok in line.split():
            if '=' not in tok:
                continue
            k, v = tok.split('=', 1)
            if '[' in k:
                base = k[:k.index('[')]
                idx  = int(k[k.index('[') + 1:k.index(']')])
                data.setdefault(base, {})[idx] = float(v)
            else:
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v

    if 'avgQueueLength' in data:
        d = data['avgQueueLength']
        ql_vals = [d[i] for i in sorted(d)]
        data['totalQL']   = sum(ql_vals)
        data['numQueues'] = len(ql_vals)

    data['permTimes'] = perm
    data['qtSeries']  = qt_series

    stable, growth_rate = _classify_stability(qt_series)
    data['unstable']   = not stable
    data['growthRate'] = growth_rate
    return data


def _classify_stability(qt_series):
    if len(qt_series) < 4:
        return True, 0.0
    times   = np.array([p[0] for p in qt_series])
    lengths = np.array([p[1] for p in qt_series])
    half = len(times) // 2
    if half < 2:
        return True, 0.0
    slope, _ = np.polyfit(times[half:], lengths[half:], 1)
    stable = slope < UNSTABLE_SLOPE_THRESHOLD
    return stable, float(slope)


def load_all(output_dir):
    rows = []
    for path in sorted(Path(output_dir).glob('out_*.txt')):
        m = _FILE_PAT.match(path.name)
        if not m:
            continue
        cfg = {
            'modality':  m.group('modality'),
            'queueType': m.group('queueType'),
            't1': float(m.group('t1')),
            't2': float(m.group('t2')),
            'k':  int(m.group('k')),
            'path': path,
        }
        try:
            cfg.update(_parse_stats(path))
        except Exception as exc:
            print(f'  skip {path.name}: {exc}')
            continue
        rows.append(cfg)
    return rows


def _dominant_qt(rows):
    cnt = defaultdict(int)
    for r in rows:
        cnt[r['queueType']] += 1
    return max(cnt, key=cnt.__getitem__, default='SERPENTINE')

# ── Styling constants ───────────────────────────────────────────────────────
_CLR = {'A': '#1f77b4', 'B': '#d62728'}
_MRK = {'A': 'o',       'B': 's'}
_LBL = {'A': 'Modalidad A', 'B': 'Modalidad B'}
_LS  = {'A': '-',           'B': '--'}


def _save(fig, name):
    p = GRAPHS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {name}')


def _add_regime_legend(ax, filled_label, hollow_label):
    """Legend with modality entries + marker-fill explanation (estable/inestable)."""
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor='#444', markeredgecolor='#444',
                          markersize=8, label=f'relleno: {filled_label}'))
    handles.append(Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor='white', markeredgecolor='#444',
                          markeredgewidth=1.6, markersize=8,
                          label=f'hueco: {hollow_label}'))
    ax.legend(handles=handles, loc='best')


def _by_modality(rows, x_key):
    by_mod = {}
    for mod in ('A', 'B'):
        pts = sorted([r for r in rows if r['modality'] == mod],
                     key=lambda r: r[x_key])
        if pts:
            by_mod[mod] = pts
    return by_mod


# ── Individual plot builders ─────────────────────────────────────────────────

def plot_queue_length(rows, x_key, x_label, tag, ylim=None):
    """Average total queue length vs x — single uniform line per modality."""
    by_mod = _by_modality(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    any_data = False
    for mod, pts in by_mod.items():
        pts_all = sorted(pts, key=lambda r: r[x_key])
        if not pts_all:
            continue
        xs = [r[x_key] for r in pts_all]
        ys = [r.get('totalQL', float('nan')) for r in pts_all]
        ax.plot(xs, ys, marker=_MRK[mod], linestyle=_LS[mod],
                color=_CLR[mod], linewidth=2.0, markersize=8,
                label=_LBL[mod])
        any_data = True
    if not any_data:
        print(f'  [{tag}] largo_fila: sin datos')
        plt.close(fig)
        return
    ax.set_xlabel(x_label)
    ax.set_ylabel('Largo promedio de fila (clientes)')
    if x_key == 'k':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    _save(fig, f'{tag}_largo_fila.png')


def plot_sensitivity(rows, x_key, x_label, tag, ylim=None):
    """Sensibilidad del largo medio de fila al parámetro X: dL̄/dX vs X.

    Se estima por diferencias finitas (central en puntos interiores,
    forward/backward en los extremos) entre las corridas disponibles. Muestra
    en qué valores del parámetro el sistema se vuelve más sensible.
    """
    by_mod = _by_modality(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    any_data = False
    for mod, pts in by_mod.items():
        pts_all = sorted(pts, key=lambda r: r[x_key])
        if len(pts_all) < 2:
            continue
        xs = np.array([r[x_key] for r in pts_all], dtype=float)
        ys = np.array([r.get('totalQL', float('nan')) for r in pts_all], dtype=float)
        dy = np.zeros_like(ys)
        dy[0]  = (ys[1]  - ys[0])  / (xs[1]  - xs[0])
        dy[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        for i in range(1, len(xs) - 1):
            dy[i] = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1])
        ax.plot(xs, dy, marker=_MRK[mod], linestyle=_LS[mod],
                color=_CLR[mod], linewidth=2.0, markersize=8, label=_LBL[mod])
        any_data = True
    if not any_data:
        print(f'  [{tag}] sensibilidad: sin datos')
        plt.close(fig)
        return
    unit = 'clientes/s' if x_key in ('t1', 't2') else 'clientes/servidor'
    sym  = {'t1': 't_1', 't2': 't_2', 'k': 'k'}[x_key]
    ax.axhline(0, color='#888', linestyle=':', linewidth=1, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(rf'$\partial\,\overline{{L}}\,/\,\partial {sym}$ ({unit})')
    if x_key == 'k':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    _save(fig, f'{tag}_sensibilidad.png')


def plot_growth_rate(rows, x_key, x_label, tag, ylim=None):
    """Growth rate (clients/s) vs x — single uniform line per modality."""
    by_mod = _by_modality(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    any_data = False
    for mod, pts in by_mod.items():
        pts_all = sorted(pts, key=lambda r: r[x_key])
        if not pts_all:
            continue
        xs = [r[x_key] for r in pts_all]
        ys = [r.get('growthRate', float('nan')) for r in pts_all]
        ax.plot(xs, ys, marker=_MRK[mod], linestyle=_LS[mod],
                color=_CLR[mod], linewidth=2.0, markersize=8,
                label=_LBL[mod])
        any_data = True
    if not any_data:
        print(f'  [{tag}] tasa_crecimiento: sin datos')
        plt.close(fig)
        return
    ax.set_xlabel(x_label)
    ax.set_ylabel('Tasa de crecimiento (clientes/s)')
    if x_key == 'k':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    _save(fig, f'{tag}_tasa_crecimiento.png')


def plot_permanence(rows, x_key, x_label, tag, ylim=None):
    """Average permanence time vs x (all runs)."""
    by_mod = _by_modality(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    any_data = False
    for mod, pts in by_mod.items():
        xs = [r[x_key] for r in pts]
        ys = [r.get('avgPermanenceTime', float('nan')) for r in pts]
        ax.plot(xs, ys,
                marker=_MRK[mod], linestyle=_LS[mod],
                color=_CLR[mod], linewidth=2.0, markersize=8,
                label=_LBL[mod])
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel(x_label)
    ax.set_ylabel('Permanencia promedio (s)')
    if x_key == 'k':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    _save(fig, f'{tag}_permanencia.png')


def plot_permanence_histogram(rows, x_key, tag, example_value):
    """Histogram of permanence times at one value of the swept parameter."""
    by_mod = _by_modality(rows, x_key)

    def pick(pts):
        cands = [r for r in pts if _approx_eq(r.get(x_key), example_value)]
        return cands[0] if cands else None

    selected = {mod: pick(pts) for mod, pts in by_mod.items()}
    any_data = any(r and r.get('permTimes') for r in selected.values())
    if not any_data:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    all_vals = []
    for mod, r in selected.items():
        if r and r.get('permTimes'):
            all_vals.extend(r['permTimes'])
    if not all_vals:
        plt.close(fig)
        return
    bins = np.linspace(min(all_vals), max(all_vals), 30)
    for mod, r in selected.items():
        if not r or not r.get('permTimes'):
            continue
        ax.hist(r['permTimes'], bins=bins,
                alpha=0.55, color=_CLR[mod], edgecolor='white', linewidth=0.4,
                label=_LBL[mod])
    ax.set_xlabel('Tiempo de permanencia (s)')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    fig.tight_layout()
    ex_str = _fmt_param(x_key, example_value)
    _save(fig, f'{tag}_hist_{x_key}={ex_str}.png')


def plot_all_histograms(rows, x_key, tag):
    """Generate one histogram per unique parameter value in the sweep."""
    values = sorted({r[x_key] for r in rows})
    for v in values:
        plot_permanence_histogram(rows, x_key, tag, v)


def _fmt_param(key, value):
    if key == 'k':
        return str(int(value))
    v = float(value)
    return f'{v:g}'


def _approx_eq(a, b, tol=0.05):
    if a is None or b is None:
        return False
    if isinstance(b, int):
        return int(round(a)) == b
    return abs(float(a) - float(b)) < tol


# ── Study driver ────────────────────────────────────────────────────────────

def _ylim_from(values, pad=0.1, floor=0.0):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    hi = max(vals)
    lo = min(vals + [floor]) if floor is not None else min(vals)
    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi != 0 else 1.0
    return (lo, hi + pad * span)


def run_study(all_rows, qt, x_key, x_label, tag, fixed_filter):
    rows = [r for r in all_rows if r['queueType'] == qt and fixed_filter(r)]
    if not rows:
        print(f'  [{tag}] sin datos')
        return

    print(f'\nEstudio {tag.replace("_", ".")}  ({len(rows)} runs)')

    perm_ylim = _ylim_from([r.get('avgPermanenceTime') for r in rows])

    for suffix, mods in (('A', ('A',)), ('B', ('B',)), ('AvB', ('A', 'B'))):
        subset = [r for r in rows if r['modality'] in mods]
        if not subset:
            continue
        sub_tag = f'{tag}_{suffix}'
        plot_queue_length(subset, x_key, x_label, sub_tag)
        plot_sensitivity(subset,  x_key, x_label, sub_tag)
        plot_permanence(subset,   x_key, x_label, sub_tag, ylim=perm_ylim)
        plot_all_histograms(subset, x_key, sub_tag)


def plot_slope_distribution(all_rows):
    """Distribución de pendientes m para justificar el umbral de estabilidad.

    Panel izquierdo: todas las corridas FREE ordenadas por pendiente (scatter),
    con el umbral y la banda "hueco natural" resaltados.
    Panel derecho: histograma de las mismas pendientes.
    """
    rows = [r for r in all_rows
            if r['queueType'] == 'FREE' and r.get('growthRate') is not None]
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r.get('growthRate', 0.0))
    slopes = np.array([r['growthRate'] for r in rows_sorted])
    mods   = np.array([r['modality']   for r in rows_sorted])

    THRESHOLD = 0.05
    GAP_LO, GAP_HI = 0.005, 0.045

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    # ── Panel izquierdo: scatter ordenado ──────────────────────────────────
    xs = np.arange(len(slopes))
    ax1.axhspan(GAP_LO, GAP_HI, color='#2ecc71', alpha=0.18,
                zorder=1, label='Hueco natural (0.005 - 0.045)')
    ax1.axhline(0.0, color='#888', linestyle=':', linewidth=1.0, zorder=1)
    ax1.axhline(THRESHOLD, color='#d62728', linestyle='--', linewidth=2.0,
                zorder=2, label=f'Umbral = {THRESHOLD} clientes/s')
    for mod in ('A', 'B'):
        mask = mods == mod
        ax1.scatter(xs[mask], slopes[mask], c=_CLR[mod], marker=_MRK[mod],
                    s=55, label=_LBL[mod], edgecolor='white',
                    linewidths=0.7, zorder=3)
    ax1.set_xlabel('Corridas ordenadas por pendiente (rank)')
    ax1.set_ylabel('Pendiente $m$ (clientes/s)')
    ax1.legend(loc='upper left', framealpha=0.9)

    # ── Panel derecho: histograma ──────────────────────────────────────────
    bins = np.linspace(slopes.min() - 0.02, slopes.max() + 0.02, 40)
    ax2.hist(slopes, bins=bins, color='#6c757d', edgecolor='white',
             linewidth=0.6)
    ax2.axvspan(GAP_LO, GAP_HI, color='#2ecc71', alpha=0.18,
                label='Hueco natural')
    ax2.axvline(THRESHOLD, color='#d62728', linestyle='--', linewidth=2.0,
                label=f'Umbral = {THRESHOLD}')
    ax2.set_xlabel('Pendiente $m$ (clientes/s)')
    ax2.set_ylabel('Cantidad de corridas')
    ax2.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    _save(fig, 'distribucion_pendientes.png')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f'Cargando desde: {OUTPUT_DIR}')
    all_rows = load_all(OUTPUT_DIR)
    if not all_rows:
        print(f'No se encontraron archivos en {OUTPUT_DIR}')
        return
    print(f'Cargados {len(all_rows)} run(s).')

    n_unstable = sum(1 for r in all_rows if r.get('unstable', False))
    print(f'  {n_unstable} run(s) clasificados como inestables.')

    qt = _dominant_qt(all_rows)
    print(f'Usando queueType = {qt}')

    EPS = 0.05

    # Estudio 2.1 — variar t2, fijo t1=1, k=5
    run_study(all_rows, qt,
              x_key='t2', x_label=r'$t_2$ (s)', tag='2_1_t1=1_k=5',
              fixed_filter=lambda r: abs(r['t1'] - 1.0) < EPS and r['k'] == 5)

    # Estudio 2.2 — variar t1, fijo t2=3, k=5
    run_study(all_rows, qt,
              x_key='t1', x_label=r'$t_1$ (s)', tag='2_2_t2=3_k=5',
              fixed_filter=lambda r: abs(r['t2'] - 3.0) < EPS and r['k'] == 5)

    # Estudio 2.3 — variar k, fijo t1=1, t2=3
    run_study(all_rows, qt,
              x_key='k', x_label=r'$k$ (servidores)', tag='2_3_t1=1_t2=3',
              fixed_filter=lambda r: abs(r['t1'] - 1.0) < EPS and abs(r['t2'] - 3.0) < EPS)

    # Estudio 2.5 — heatmap t1 × t2, k=5
    heatmap_dir = REPO_ROOT / 'tp3-output-2_5'
    if heatmap_dir.is_dir():
        run_heatmaps(heatmap_dir, qt)

    # Diagnóstico del umbral de estabilidad
    plot_slope_distribution(all_rows)

    print('\nListo.')


# ── Heatmap helpers (2.5) ──────────────────────────────────────────────────

from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patheffects as pe

def run_heatmaps(output_dir, qt):
    rows = load_all(output_dir)
    rows = [r for r in rows if r['queueType'] == qt and r['k'] == 5]
    if not rows:
        print('  [2.5] sin datos para heatmap')
        return
    print(f'\nEstudio 2.5  ({len(rows)} runs)')

    data = {}
    for mod in ('A', 'B'):
        mod_rows = [r for r in rows if r['modality'] == mod]
        if mod_rows:
            data[mod] = mod_rows
    if not data:
        return

    _heatmap_stability(data)
    _heatmap_length_vs_permanence(data)


def _build_grid(rows):
    t1_vals = sorted(set(r['t1'] for r in rows))
    t2_vals = sorted(set(r['t2'] for r in rows))
    lookup = {}
    for r in rows:
        lookup[(r['t1'], r['t2'])] = r
    return t1_vals, t2_vals, lookup


def _cell_edges(vals):
    edges = []
    for i, v in enumerate(vals):
        if i == 0:
            lo = v - (vals[1] - v) / 2 if len(vals) > 1 else v - 0.5
        else:
            lo = (vals[i - 1] + v) / 2
        edges.append(lo)
    last = vals[-1]
    if len(vals) > 1:
        edges.append(last + (last - vals[-2]) / 2)
    else:
        edges.append(last + 0.5)
    return np.array(edges)


def _text_color(val, vmax):
    return 'white' if val > vmax * 0.55 else '#222222'


def _draw_rho_line(ax, t1_vals):
    t1_lo = t1_vals[0] * 0.7
    t1_hi = t1_vals[-1] * 1.1
    t1s = np.linspace(t1_lo, t1_hi, 200)
    ax.plot(t1s, 5.0 * t1s, color='#222222', linewidth=2.2, linestyle='--',
            zorder=5, label=r'$\rho = 1$')
    ax.legend(loc='upper left', framealpha=0.85, edgecolor='none',
              fontsize=11, handlelength=1.8)


def _annotate_cell(ax, x, y, text, val, vmax, bold=False):
    txt = ax.text(x, y, text, ha='center', va='center',
                  fontsize=10, fontweight='bold' if bold else 'medium',
                  color=_text_color(val, vmax), zorder=10)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.4)])


def _hatch_unstable(ax, t1, t2, t1_edges, t2_edges, j, i):
    x0 = t1_edges[j]
    y0 = t2_edges[i]
    w = t1_edges[j + 1] - x0
    h = t2_edges[i + 1] - y0
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                            hatch='///', edgecolor='#555555', linewidth=0,
                            alpha=0.35, zorder=4))


def _style_ax(ax, t1_vals, t2_vals, subtitle):
    ax.set_xlabel(r'$t_1$ (s)', fontsize=13)
    ax.set_ylabel(r'$t_2$ (s)', fontsize=13)
    ax.set_xticks(t1_vals)
    ax.set_yticks(t2_vals)
    ax.set_xlim(_cell_edges(t1_vals)[0], _cell_edges(t1_vals)[-1])
    ax.set_ylim(_cell_edges(t2_vals)[0], _cell_edges(t2_vals)[-1])
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=13, fontweight='semibold')


def _heatmap_stability(data):
    """Binary stability map: green = stable, red = unstable."""
    from matplotlib.colors import ListedColormap

    ref_rows = list(data.values())[0]
    t1_vals, t2_vals, _ = _build_grid(ref_rows)
    t1_edges = _cell_edges(t1_vals)
    t2_edges = _cell_edges(t2_vals)

    cmap_bin = ListedColormap(['#2ecc71', '#e74c3c'])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)

    for idx, (mod, mod_rows) in enumerate(data.items()):
        _, _, lookup = _build_grid(mod_rows)
        Z = np.full((len(t2_vals), len(t1_vals)), np.nan)
        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                r = lookup.get((t1, t2))
                if r is None:
                    continue
                Z[i, j] = 1.0 if r.get('unstable', False) else 0.0

        ax = axes[idx]
        ax.pcolormesh(t1_edges, t2_edges, Z, cmap=cmap_bin,
                      vmin=0, vmax=1, shading='flat', zorder=1)

        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                r = lookup.get((t1, t2))
                if r is None:
                    continue
                unstable = r.get('unstable', False)
                rho = t2 / (5.0 * t1)
                ax.text(t1, t2, f'$\\rho$={rho:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='medium', color='white',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black', alpha=0.3)])

        _style_ax(ax, t1_vals, t2_vals, f'Modalidad {mod}')
        if idx > 0:
            ax.set_ylabel('')

    # Custom legend instead of colorbar
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Estable'),
                       Patch(facecolor='#e74c3c', label='Inestable')]
    fig.legend(handles=legend_elements, loc='center right', fontsize=13,
               framealpha=0.9, edgecolor='none', borderpad=1.0)
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _save(fig, '2_5_k=5_heatmap_estabilidad_AvB.png')


def _heatmap_length_vs_permanence(data):
    """2x2 heatmap: largo promedio (fila sup.) y permanencia (fila inf.), A vs B."""
    ref_rows = list(data.values())[0]
    t1_vals, t2_vals, _ = _build_grid(ref_rows)
    t1_edges = _cell_edges(t1_vals)
    t2_edges = _cell_edges(t2_vals)

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0))

    # ── Row 0: largo promedio ──────────────────────────────────────────────
    vmin_ql, vmax_ql = np.inf, 0
    ql_grids = {}
    for mod, mod_rows in data.items():
        _, _, lookup = _build_grid(mod_rows)
        Z = np.full((len(t2_vals), len(t1_vals)), np.nan)
        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                r = lookup.get((t1, t2))
                if r is None:
                    continue
                val = r.get('totalQL', np.nan)
                if not np.isnan(val):
                    Z[i, j] = val
        ql_grids[mod] = (Z, lookup)
        if not np.all(np.isnan(Z)):
            vmax_ql = max(vmax_ql, np.nanmax(Z))
            pos = Z[Z > 0]
            if pos.size:
                vmin_ql = min(vmin_ql, float(np.nanmin(pos)))
    if not np.isfinite(vmin_ql):
        vmin_ql = 0.1
    vmin_ql = max(vmin_ql, 0.1)

    norm_ql = LogNorm(vmin=vmin_ql, vmax=max(vmax_ql, vmin_ql * 10))
    for idx, (mod, (Z, lookup)) in enumerate(ql_grids.items()):
        ax = axes[0, idx]
        Z_draw = np.where(np.isnan(Z) | (Z <= 0), vmin_ql, Z)
        im_ql = ax.pcolormesh(t1_edges, t2_edges, Z_draw,
                              cmap='YlGnBu', norm=norm_ql,
                              shading='flat', zorder=2)
        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                val = Z[i, j]
                if np.isnan(val):
                    continue
                _annotate_cell(ax, t1, t2, f'{val:.1f}', val, vmax_ql)
        _style_ax(ax, t1_vals, t2_vals, f'Largo promedio — Modalidad {mod}')
        if idx > 0:
            ax.set_ylabel('')

    # Se completa abajo con add_axes para evitar superposición.
    _im_ql_ref = im_ql

    # ── Row 1: permanencia promedio ────────────────────────────────────────
    vmax_pm = 0
    pm_grids = {}
    for mod, mod_rows in data.items():
        _, _, lookup = _build_grid(mod_rows)
        Z = np.full((len(t2_vals), len(t1_vals)), np.nan)
        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                r = lookup.get((t1, t2))
                if r is None:
                    continue
                Z[i, j] = r.get('avgPermanenceTime', np.nan)
        pm_grids[mod] = (Z, lookup)
        if not np.all(np.isnan(Z)):
            vmax_pm = max(vmax_pm, np.nanmax(Z))

    norm_pm = Normalize(vmin=0, vmax=max(vmax_pm, 1.0))
    for idx, (mod, (Z, lookup)) in enumerate(pm_grids.items()):
        ax = axes[1, idx]
        im_pm = ax.pcolormesh(t1_edges, t2_edges, Z, cmap='inferno_r',
                              norm=norm_pm, shading='flat', zorder=2)
        for j, t1 in enumerate(t1_vals):
            for i, t2 in enumerate(t2_vals):
                val = Z[i, j]
                if np.isnan(val):
                    continue
                _annotate_cell(ax, t1, t2, f'{val:.0f}', val, vmax_pm)
        _style_ax(ax, t1_vals, t2_vals, f'Permanencia — Modalidad {mod}')
        if idx > 0:
            ax.set_ylabel('')

    # ── Layout y colorbars fuera de los axes ───────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.86, top=0.95, bottom=0.06,
                        hspace=0.25, wspace=0.10)
    cbar_ax_top = fig.add_axes([0.89, 0.55, 0.018, 0.38])
    cbar_ql = fig.colorbar(_im_ql_ref, cax=cbar_ax_top)
    cbar_ql.set_label('Largo promedio de fila (clientes, escala log)',
                      fontsize=12)

    cbar_ax_bot = fig.add_axes([0.89, 0.08, 0.018, 0.38])
    cbar_pm = fig.colorbar(im_pm, cax=cbar_ax_bot)
    cbar_pm.set_label('Permanencia promedio (s)', fontsize=12)

    _save(fig, '2_5_k=5_heatmap_fila_y_permanencia_AvB.png')


if __name__ == '__main__':
    main()
