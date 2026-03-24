# VISUALIZADOR TP2 (versión modificada, mantiene estructura original)
# - Todo en español
# - GIF combinado (movimiento + polarización)
# - NO requiere argumentos obligatorios (usa defaults como antes)
# - Mantiene PNG final de polarización

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Frame:
    step: int
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    leader_mask: np.ndarray


def parse_step(comment_line: str) -> int:
    match = re.search(r"-?\d+", comment_line)
    return int(match.group(0)) if match else 0


def parse_frame_block(lines: list[str], n_particles: int) -> np.ndarray:
    data = []
    for line in lines:
        parts = line.strip().split()
        # esperamos exactamente 6 columnas
        if len(parts) != 6:
            continue
        try:
            row = [float(p) for p in parts]
            data.append(row)
        except ValueError:
            continue

    if len(data) != n_particles:
        raise ValueError(f"Se esperaban {n_particles} partículas, pero se leyeron {len(data)}")

    return np.array(data)


def load_frames(path: Path, stride: int, max_frames: int | None) -> list[Frame]:
    frames: list[Frame] = []
    frame_idx = 0

    with path.open("r", encoding="utf-8") as f:

        # 🔴 leer header UNA sola vez
        header = f.readline()
        n_particles = int(header.split()[0])

        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip()

            # buscamos inicio de frame
            if not line.startswith("step"):
                continue

            comment = line  # "step X"

            # leer partículas
            particle_lines = []
            while len(particle_lines) < n_particles:
                l = f.readline()
                if not l:
                    break

                l = l.strip()

                if l == "" or l.startswith("step"):
                    continue

                particle_lines.append(l)

            if len(particle_lines) != n_particles:
                break  # frame incompleto → cortar

            if frame_idx % stride == 0:
                data = parse_frame_block(particle_lines, n_particles)

                x, y = data[:, 1], data[:, 2]
                vx, vy = data[:, 3], data[:, 4]
                leader_mask = data[:, 5] > 0.5

                frames.append(Frame(parse_step(comment), x, y, vx, vy, leader_mask))

                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1

    return frames

def compute_polarization(frame: Frame) -> float:
    sum_vx = float(np.sum(frame.vx))
    sum_vy = float(np.sum(frame.vy))
    return math.hypot(sum_vx, sum_vy) / len(frame.vx)


def make_combined_animation(frames: list[Frame], fps: int, gif_path: Path, dpi: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Más espacio entre los dos paneles
    fig.subplots_adjust(wspace=0.35)

    # ----- MOVIMIENTO -----
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")

    norm = plt.Normalize(0, 2 * math.pi)
    cmap = plt.get_cmap("hsv")

    f0 = frames[0]
    angles = np.mod(np.arctan2(f0.vy, f0.vx), 2 * math.pi)

    no_lider = ~f0.leader_mask

    q_no_lider = ax1.quiver(
        f0.x[no_lider],
        f0.y[no_lider],
        f0.vx[no_lider],
        f0.vy[no_lider],
        angles[no_lider],
        cmap=cmap,
        norm=norm
    )

    q_lider = ax1.quiver(
        f0.x[f0.leader_mask],
        f0.y[f0.leader_mask],
        f0.vx[f0.leader_mask],
        f0.vy[f0.leader_mask],
        color="#d62828"
    )

    puntos_lider = ax1.scatter(
        f0.x[f0.leader_mask],
        f0.y[f0.leader_mask],
        s=40,
        c="#d62828",
        edgecolors="#1b1b1b",
        label="Líder"
    )

    if np.any(f0.leader_mask):
        ax1.legend(loc="upper right", fontsize=8)

    cbar = fig.colorbar(q_no_lider, ax=ax1)
    cbar.set_label("velocidad angular (rad)")

    # ----- POLARIZACIÓN -----
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    ax2.set_xlabel("tiempo (steps)")
    ax2.set_ylabel("polarización (va)")
    ax2.set_title("Polarización vs tiempo")
    ax2.set_xlim(steps.min(), steps.max())
    ax2.set_ylim(0, 1.05)

    line, = ax2.plot([], [])
    point = ax2.scatter([], [])

    def update(i):
        f = frames[i]
        angles = np.mod(np.arctan2(f.vy, f.vx), 2 * math.pi)

        no_lider = ~f.leader_mask

        q_no_lider.set_offsets(np.column_stack((f.x[no_lider], f.y[no_lider])))
        q_no_lider.set_UVC(f.vx[no_lider], f.vy[no_lider], angles[no_lider])

        q_lider.set_offsets(np.column_stack((f.x[f.leader_mask], f.y[f.leader_mask])))
        q_lider.set_UVC(f.vx[f.leader_mask], f.vy[f.leader_mask])

        puntos_lider.set_offsets(np.column_stack((f.x[f.leader_mask], f.y[f.leader_mask])))

        line.set_data(steps[:i+1], va[:i+1])
        point.set_offsets([[steps[i], va[i]]])

        return q_no_lider, q_lider, puntos_lider, line, point

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_polarization_png(frames: list[Frame], path: Path):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    plt.figure()
    plt.plot(steps, va)
    plt.xlabel("tiempo (steps)")
    plt.ylabel("polarización (va)")
    plt.title("Polarización vs tiempo")

    # Ajustes para que el 0 quede pegado a los ejes
    plt.xlim(steps.min(), steps.max())
    plt.ylim(0.0, 1.05)
    plt.margins(x=0, y=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description="Visualizador TP2")
    parser.add_argument("--input", type=Path, default=repo_root / "tp2-output" / "output.xyz")
    parser.add_argument("--gif", type=Path, default=repo_root / "tp2-visual" / "tp2_combined.gif")
    parser.add_argument("--png", type=Path, default=repo_root / "tp2-visual" / "polarization_final.png")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=2000)

    args = parser.parse_args()

    frames = load_frames(args.input, args.stride, args.max_frames)

    make_combined_animation(frames, args.fps, args.gif, 120)
    save_polarization_png(frames, args.png)

    print(f"GIF generado: {args.gif}")
    print(f"PNG generado: {args.png}")


if __name__ == "__main__":
    main()
