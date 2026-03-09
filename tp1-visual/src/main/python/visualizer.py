import argparse
import os
from pathlib import Path
import sys

try:
    import matplotlib
except ModuleNotFoundError as exc:
    sys.exit(
        "Matplotlib is not installed. Install it with 'python3 -m pip install matplotlib' and rerun."
    )

if "MPLBACKEND" not in os.environ:
    for backend in ("TkAgg", "Qt5Agg", "Agg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Particle Simulation Visualizer")
parser.add_argument("--rc", type=float, default=None,
                    help="Detection radius (overrides value in static.txt)")
default_bin = Path(__file__).resolve().parents[4] / "tp1-output"
parser.add_argument("--input-dir", type=Path, default=default_bin,
                    help="Directory containing static.txt, dynamic.txt, neighbors.txt")
args = parser.parse_args()

BIN_PATH = args.input_dir.expanduser().resolve()
if not BIN_PATH.exists():
    sys.exit(f"Input directory '{BIN_PATH}' not found. Generate data with the Java project first.")

# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_static(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    N = int(lines[0])
    L = float(lines[1])
    radii = [float(lines[2 + i]) for i in range(N)]
    rc = float(lines[2 + N]) if len(lines) > 2 + N else None
    return N, L, radii, rc

def parse_dynamic(path):
    """Returns list of (time, [(x, y, vx, vy), ...]) frames."""
    frames = []
    with open(path) as f:
        lines = [l.strip() for l in f]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) == 1:
            # Time step header
            t = float(parts[0])
            particles = []
            i += 1
            while i < len(lines):
                pline = lines[i].strip()
                if not pline or len(pline.split()) == 1:
                    break
                px, py, pvx, pvy = map(float, pline.split())
                particles.append((px, py, pvx, pvy))
                i += 1
            frames.append((t, particles))
        else:
            i += 1
    return frames

def parse_neighbors(path):
        """Returns dict: particle_index -> [neighbor_indices]"""
        neighbors = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                idx = int(parts[0])
                nbrs = [int(p) for p in parts[1:]] if len(parts) > 1 else []
                neighbors[idx] = nbrs
        return neighbors

# ── Load data ──────────────────────────────────────────────────────────────────

static_path   = BIN_PATH / "static.txt"
dynamic_path  = BIN_PATH / "dynamic.txt"
neighbor_path = BIN_PATH / "neighbors.txt"

N, L, radii, rc = parse_static(static_path)
if args.rc is not None:
    rc = args.rc
frames           = parse_dynamic(dynamic_path)
neighbors        = parse_neighbors(neighbor_path)

# Use first frame
_t0, particles = frames[0]
xs = [p[0] for p in particles]
ys = [p[1] for p in particles]

# ── Figure setup ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect("equal")
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

title = f"Particle Simulation  —  N={N}  L={L}"
if rc is not None:
    title += f"  rc={rc}"
ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)

# Boundary box
boundary = patches.Rectangle((0, 0), L, L,
                             linewidth=1.5, edgecolor="#30363d",
                             facecolor="none", zorder=1)
ax.add_patch(boundary)

# ── Draw particles ─────────────────────────────────────────────────────────────

BASE_COLOR   = "#58a6ff"
BASE_EDGE    = "#1f6feb"
SELECT_COLOR = "#ff4444"
SELECT_EDGE  = "#cc0000"
NEIGHBOR_COLOR = "#3fb950"
NEIGHBOR_EDGE  = "#238636"
RC_COLOR     = "#ff8888"

particle_circles = []
for i in range(N):
    c = patches.Circle((xs[i], ys[i]), radii[i],
                       facecolor=BASE_COLOR, edgecolor=BASE_EDGE,
                       linewidth=0.8, alpha=0.85, zorder=2, picker=True)
    ax.add_patch(c)
    particle_circles.append(c)

# ── State & interaction ────────────────────────────────────────────────────────

selected         = [None]   # mutable for closure
overlay_patches  = []       # extra patches to remove on deselect

def clear_overlays():
    for p in overlay_patches:
        p.remove()
    overlay_patches.clear()

def reset_colors():
    for c in particle_circles:
        c.set_facecolor(BASE_COLOR)
        c.set_edgecolor(BASE_EDGE)
        c.set_zorder(2)

def on_pick(event):
    if not isinstance(event.artist, patches.Circle):
        return
    try:
        idx = particle_circles.index(event.artist)
    except ValueError:
        return

    clear_overlays()
    reset_colors()

    # Toggle deselect
    if selected[0] == idx:
        selected[0] = None
        fig.canvas.draw_idle()
        return

    selected[0] = idx

    # Selected particle → red
    particle_circles[idx].set_facecolor(SELECT_COLOR)
    particle_circles[idx].set_edgecolor(SELECT_EDGE)
    particle_circles[idx].set_zorder(5)

    # rc ring (only if rc was provided)
    if rc is not None:
        rc_ring = patches.Circle((xs[idx], ys[idx]), rc + radii[idx],
                                 linewidth=1.2, linestyle="--",
                                 edgecolor=RC_COLOR, facecolor="none",
                                 alpha=0.6, zorder=4)
        ax.add_patch(rc_ring)
        overlay_patches.append(rc_ring)

    # Neighbor highlights → green overlay circles
    for nbr_id in neighbors.get(idx, []):
        nbr = nbr_id
        if 0 <= nbr < N:
            p = patches.Circle((xs[nbr], ys[nbr]), radii[nbr],
                               facecolor=NEIGHBOR_COLOR, edgecolor=NEIGHBOR_EDGE,
                               linewidth=0.8, alpha=0.9, zorder=3)
            ax.add_patch(p)
            overlay_patches.append(p)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("pick_event", on_pick)

plt.tight_layout()
plt.show()
