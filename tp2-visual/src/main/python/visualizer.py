# VISUALIZADOR TP2
# - GIF de particulas (particles.gif)
# - GIF y PNG de polarización vs tiempo (polarization_time.gif/.png)
# - Modo comparacion: --compare RUTA ETA (repetible)
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


STATIONARY_REPEAT_COUNT = 5
STATIONARY_TOLERANCE = 5e-2
STATIONARY_COLOR = "#d62828"
STATIONARY_SHADE_ALPHA = 0.16
ANALYSIS_STATIONARY_START_FRACTION = 0.4


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


def detect_stationary_start_index_by_window(
    values: np.ndarray,
    repeat_count: int = STATIONARY_REPEAT_COUNT,
    tolerance: float = STATIONARY_TOLERANCE,
) -> tuple[int | None, int | None]:
    if repeat_count <= 0 or values.size == 0:
        return None, None
    if repeat_count == 1:
        return 0, 0
    if values.size < repeat_count:
        return None, None

    for start in range(0, values.size - repeat_count + 1):
        window = values[start : start + repeat_count]
        if float(np.max(window) - np.min(window)) <= tolerance:
            return start, start + repeat_count - 1

    return None, None


def detect_stationary_start_index_like_analysis(
    steps: np.ndarray,
    start_fraction: float = ANALYSIS_STATIONARY_START_FRACTION,
) -> tuple[int | None, int | None]:
    if steps.size == 0:
        return None, None

    max_step = int(np.max(steps))
    if steps.size >= 2:
        diffs = np.diff(np.sort(steps))
        diffs = diffs[diffs > 0]
        sample_step = int(round(float(np.median(diffs)))) if diffs.size > 0 else 1
    else:
        sample_step = 1
    sample_step = max(sample_step, 1)

    estimated_total_steps = max_step + sample_step
    threshold = int(start_fraction * estimated_total_steps)

    candidates = np.where(steps > threshold)[0]
    if candidates.size == 0:
        return None, None

    idx = int(candidates[0])
    return idx, idx


def detect_stationary_start_index(
    steps: np.ndarray,
    values: np.ndarray,
    repeat_count: int = STATIONARY_REPEAT_COUNT,
    tolerance: float = STATIONARY_TOLERANCE,
    analysis_start_fraction: float = ANALYSIS_STATIONARY_START_FRACTION,
) -> tuple[int | None, int | None]:
    start_idx, confirm_idx = detect_stationary_start_index_by_window(values, repeat_count, tolerance)
    if start_idx is not None:
        return start_idx, confirm_idx

    return detect_stationary_start_index_like_analysis(steps, analysis_start_fraction)


def mark_stationary_step_on_xaxis(ax: plt.Axes, step_value: float) -> None:
    ticks = np.asarray(ax.get_xticks(), dtype=float)
    if ticks.size == 0:
        ticks = np.array([step_value], dtype=float)
    elif not np.any(np.isclose(ticks, step_value, atol=1e-12, rtol=0.0)):
        ticks = np.sort(np.append(ticks, step_value))

    ax.set_xticks(ticks)

    labels: list[str] = []
    for tick in ticks:
        if np.isclose(tick, step_value, atol=1e-12, rtol=0.0):
            labels.append(f"{int(round(step_value))}\n(estado est.)")
        elif float(tick).is_integer():
            labels.append(str(int(tick)))
        else:
            labels.append(f"{tick:g}")
    ax.set_xticklabels(labels)

    for tick, tick_label in zip(ticks, ax.get_xticklabels()):
        if np.isclose(tick, step_value, atol=1e-12, rtol=0.0):
            tick_label.set_color(STATIONARY_COLOR)
            tick_label.set_fontweight("bold")
            tick_label.set_y(-0.04)
            tick_label.set_va("top")


def infer_box_size(frames: list[Frame]) -> float:
    box_size = 0.0
    for frame in frames:
        if frame.x.size > 0:
            box_size = max(box_size, float(np.max(frame.x)))
        if frame.y.size > 0:
            box_size = max(box_size, float(np.max(frame.y)))
    return box_size * 1.02 if box_size > 0 else 1.0


def make_particles_animation(frames: list[Frame], fps: int, gif_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(top=0.9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    f0 = frames[0]
    box_size = infer_box_size(frames)
    ax.set_xlim(0.0, box_size)
    ax.set_ylim(0.0, box_size)

    norm = plt.Normalize(0, 2 * math.pi)
    cmap = plt.get_cmap("hsv")

    angles = np.mod(np.arctan2(f0.vy, f0.vx), 2 * math.pi)
    no_lider = ~f0.leader_mask
    has_no_lider = bool(np.any(no_lider))

    if has_no_lider:
        q_no_lider = ax.quiver(
            f0.x[no_lider],
            f0.y[no_lider],
            f0.vx[no_lider],
            f0.vy[no_lider],
            angles[no_lider],
            cmap=cmap,
            norm=norm,
        )
    else:
        q_no_lider = ax.quiver([], [], [], [], color="#4b9be0")

    has_lider = bool(np.any(f0.leader_mask))
    q_lider = None
    puntos_lider = None
    if has_lider:
        q_lider = ax.quiver(
            f0.x[f0.leader_mask],
            f0.y[f0.leader_mask],
            f0.vx[f0.leader_mask],
            f0.vy[f0.leader_mask],
            color="#d62828",
        )

        puntos_lider = ax.scatter(
            f0.x[f0.leader_mask],
            f0.y[f0.leader_mask],
            s=40,
            c="#d62828",
            edgecolors="#1b1b1b",
            label="Lider",
        )

        ax.legend(loc="upper right", fontsize=8)

    if has_no_lider:
        cbar = fig.colorbar(q_no_lider, ax=ax)
        cbar.set_label("velocidad angular (rad)")

    step_text = ax.text(
        0.0,
        1.02,
        f"step = {f0.step}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"},
    )

    def update(i):
        f = frames[i]
        angles_i = np.mod(np.arctan2(f.vy, f.vx), 2 * math.pi)
        no_lider_i = ~f.leader_mask

        if has_no_lider:
            q_no_lider.set_offsets(np.column_stack((f.x[no_lider_i], f.y[no_lider_i])))
            q_no_lider.set_UVC(f.vx[no_lider_i], f.vy[no_lider_i], angles_i[no_lider_i])
        else:
            q_no_lider.set_offsets(np.empty((0, 2)))
            q_no_lider.set_UVC(np.array([]), np.array([]))

        if q_lider is not None and puntos_lider is not None:
            q_lider.set_offsets(np.column_stack((f.x[f.leader_mask], f.y[f.leader_mask])))
            q_lider.set_UVC(f.vx[f.leader_mask], f.vy[f.leader_mask])
            puntos_lider.set_offsets(np.column_stack((f.x[f.leader_mask], f.y[f.leader_mask])))
        step_text.set_text(f"step = {f.step}")

        artists = [q_no_lider, step_text]
        if q_lider is not None and puntos_lider is not None:
            artists.extend([q_lider, puntos_lider])
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def make_combined_animation(frames: list[Frame], fps: int, gif_path: Path, dpi: int):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    fig, (ax_particles, ax_va) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.25)

    f0 = frames[0]
    box_size = infer_box_size(frames)
    ax_particles.set_xlabel("x (m)")
    ax_particles.set_ylabel("y (m)")
    ax_particles.set_aspect("equal")
    ax_particles.set_xlim(0.0, box_size)
    ax_particles.set_ylim(0.0, box_size)

    norm = plt.Normalize(0, 2 * math.pi)
    cmap = plt.get_cmap("hsv")
    angles = np.mod(np.arctan2(f0.vy, f0.vx), 2 * math.pi)
    no_lider = ~f0.leader_mask
    has_no_lider = bool(np.any(no_lider))

    if has_no_lider:
        q_no_lider = ax_particles.quiver(
            f0.x[no_lider],
            f0.y[no_lider],
            f0.vx[no_lider],
            f0.vy[no_lider],
            angles[no_lider],
            cmap=cmap,
            norm=norm,
        )
    else:
        q_no_lider = ax_particles.quiver([], [], [], [], color="#4b9be0")

    has_lider = bool(np.any(f0.leader_mask))
    q_lider = None
    puntos_lider = None
    if has_lider:
        q_lider = ax_particles.quiver(
            f0.x[f0.leader_mask],
            f0.y[f0.leader_mask],
            f0.vx[f0.leader_mask],
            f0.vy[f0.leader_mask],
            color="#d62828",
        )
        puntos_lider = ax_particles.scatter(
            f0.x[f0.leader_mask],
            f0.y[f0.leader_mask],
            s=40,
            c="#d62828",
            edgecolors="#1b1b1b",
            label="Lider",
        )
        ax_particles.legend(loc="upper right", fontsize=8)

    if has_no_lider:
        cbar = fig.colorbar(q_no_lider, ax=ax_particles)
        cbar.set_label("velocidad angular (rad)")

    step_text = ax_particles.text(
        0.0,
        1.02,
        f"step = {f0.step}",
        transform=ax_particles.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"},
    )

    ax_va.set_xlabel("tiempo (steps)")
    ax_va.set_ylabel("polarización (va)")
    ax_va.set_title("Polarización vs tiempo")
    if len(steps) == 1:
        ax_va.set_xlim(float(steps[0]) - 1.0, float(steps[0]) + 1.0)
    else:
        ax_va.set_xlim(float(steps.min()), float(steps.max()))
    ax_va.set_ylim(0.0, 1.05)
    ax_va.grid(True, alpha=0.3)

    stationary_start_idx, stationary_confirm_idx = detect_stationary_start_index(steps, va)
    stationary_step = float(steps[stationary_start_idx]) if stationary_start_idx is not None else None
    stationary_va = float(va[stationary_start_idx]) if stationary_start_idx is not None else None

    stationary_span = None
    stationary_line = None
    stationary_start_point = None
    if stationary_step is not None and stationary_va is not None:
        x_min, x_max = ax_va.get_xlim()
        if x_min <= stationary_step <= x_max:
            mark_stationary_step_on_xaxis(ax_va, stationary_step)
            stationary_span = ax_va.axvspan(
                x_min,
                stationary_step,
                color=STATIONARY_COLOR,
                alpha=STATIONARY_SHADE_ALPHA,
                zorder=0,
            )
            stationary_line = ax_va.axvline(
                stationary_step,
                color=STATIONARY_COLOR,
                linewidth=2.0,
                alpha=0.95,
                zorder=4,
            )
            stationary_start_point = ax_va.scatter(
                [stationary_step],
                [stationary_va],
                s=68,
                c=STATIONARY_COLOR,
                edgecolors="#1b1b1b",
                linewidths=0.6,
                zorder=5,
            )
            stationary_span.set_visible(False)
            stationary_line.set_visible(False)
            stationary_start_point.set_visible(False)

    line, = ax_va.plot([], [], color="#0a58ca", linewidth=2.0)
    point = ax_va.scatter([], [], s=40, color="#d62828")
    label = ax_va.text(
        0.0,
        1.10,
        "",
        transform=ax_va.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"},
    )

    def update(i):
        frame = frames[i]
        angles_i = np.mod(np.arctan2(frame.vy, frame.vx), 2 * math.pi)
        no_lider_i = ~frame.leader_mask

        if has_no_lider:
            q_no_lider.set_offsets(np.column_stack((frame.x[no_lider_i], frame.y[no_lider_i])))
            q_no_lider.set_UVC(frame.vx[no_lider_i], frame.vy[no_lider_i], angles_i[no_lider_i])
        else:
            q_no_lider.set_offsets(np.empty((0, 2)))
            q_no_lider.set_UVC(np.array([]), np.array([]))

        if q_lider is not None and puntos_lider is not None:
            q_lider.set_offsets(np.column_stack((frame.x[frame.leader_mask], frame.y[frame.leader_mask])))
            q_lider.set_UVC(frame.vx[frame.leader_mask], frame.vy[frame.leader_mask])
            puntos_lider.set_offsets(np.column_stack((frame.x[frame.leader_mask], frame.y[frame.leader_mask])))

        step_text.set_text(f"step = {frame.step}")

        line.set_data(steps[: i + 1], va[: i + 1])
        point.set_offsets(np.array([[steps[i], va[i]]]))
        label.set_text(f"step: {int(steps[i])} | polarización: {va[i]:.4f}")

        if (
            stationary_span is not None
            and stationary_line is not None
            and stationary_start_point is not None
            and stationary_confirm_idx is not None
        ):
            is_visible = i >= stationary_confirm_idx
            stationary_span.set_visible(is_visible)
            stationary_line.set_visible(is_visible)
            stationary_start_point.set_visible(is_visible)

        artists = [q_no_lider, step_text, line, point, label]
        if q_lider is not None and puntos_lider is not None:
            artists.extend([q_lider, puntos_lider])
        if (
            stationary_span is not None
            and stationary_line is not None
            and stationary_start_point is not None
        ):
            artists.extend([stationary_span, stationary_line, stationary_start_point])
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def make_polarization_animation(frames: list[Frame], fps: int, gif_path: Path, dpi: int):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.87, bottom=0.25)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarización (va)")
    ax.set_title("Polarización vs tiempo")

    if len(steps) == 1:
        ax.set_xlim(float(steps[0]) - 1.0, float(steps[0]) + 1.0)
    else:
        ax.set_xlim(float(steps.min()), float(steps.max()))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    stationary_start_idx, stationary_confirm_idx = detect_stationary_start_index(steps, va)
    stationary_step = float(steps[stationary_start_idx]) if stationary_start_idx is not None else None
    stationary_va = float(va[stationary_start_idx]) if stationary_start_idx is not None else None

    stationary_span = None
    stationary_line = None
    stationary_start_point = None
    if stationary_step is not None and stationary_va is not None:
        x_min, x_max = ax.get_xlim()
        if x_min <= stationary_step <= x_max:
            mark_stationary_step_on_xaxis(ax, stationary_step)
            stationary_span = ax.axvspan(
                x_min,
                stationary_step,
                color=STATIONARY_COLOR,
                alpha=STATIONARY_SHADE_ALPHA,
                zorder=0,
            )
            stationary_line = ax.axvline(
                stationary_step,
                color=STATIONARY_COLOR,
                linewidth=2.0,
                alpha=0.95,
                zorder=4,
            )
            stationary_start_point = ax.scatter(
                [stationary_step],
                [stationary_va],
                s=68,
                c=STATIONARY_COLOR,
                edgecolors="#1b1b1b",
                linewidths=0.6,
                zorder=5,
            )
            stationary_span.set_visible(False)
            stationary_line.set_visible(False)
            stationary_start_point.set_visible(False)

    line, = ax.plot([], [], color="#0a58ca", linewidth=2.0)
    point = ax.scatter([], [], s=40, color="#d62828")
    label = ax.text(
        0.0,
        1.02,
        "",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"},
    )

    def update(i):
        line.set_data(steps[: i + 1], va[: i + 1])
        point.set_offsets(np.array([[steps[i], va[i]]]))
        label.set_text(f"step: {int(steps[i])} | polarización: {va[i]:.4f}")

        artists = [line, point, label]
        if (
            stationary_span is not None
            and stationary_line is not None
            and stationary_start_point is not None
            and stationary_confirm_idx is not None
        ):
            is_visible = i >= stationary_confirm_idx
            stationary_span.set_visible(is_visible)
            stationary_line.set_visible(is_visible)
            stationary_start_point.set_visible(is_visible)
            artists.extend([stationary_span, stationary_line, stationary_start_point])

        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_polarization_png(frames: list[Frame], path: Path):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25)
    ax.plot(steps, va, color="#0a58ca", linewidth=2.0)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarización (va)")
    ax.set_title("Polarización vs tiempo")
    ax.set_xlim(float(np.min(steps)), float(np.max(steps)))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0, y=0)

    stationary_start_idx, _ = detect_stationary_start_index(steps, va)
    if stationary_start_idx is not None:
        stationary_step = float(steps[stationary_start_idx])
        stationary_va = float(va[stationary_start_idx])
        x_min, x_max = ax.get_xlim()
        if x_min <= stationary_step <= x_max:
            mark_stationary_step_on_xaxis(ax, stationary_step)
            ax.axvspan(
                x_min,
                stationary_step,
                color=STATIONARY_COLOR,
                alpha=STATIONARY_SHADE_ALPHA,
                zorder=0,
            )
            ax.axvline(
                stationary_step,
                color=STATIONARY_COLOR,
                linewidth=2.0,
                alpha=0.95,
                zorder=4,
            )
            ax.scatter(
                [stationary_step],
                [stationary_va],
                s=68,
                c=STATIONARY_COLOR,
                edgecolors="#1b1b1b",
                linewidths=0.6,
                zorder=5,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _eta_legend(eta: float) -> str:
    return f"η = {eta:g}"


def save_polarization_overlay(
    series: list[tuple[Path, float]],
    stride: int,
    max_frames: int | None,
    path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, (xyz, eta) in enumerate(series):
        frames = load_frames(xyz, stride, max_frames)
        if not frames:
            continue
        steps = np.array([f.step for f in frames])
        va = np.array([compute_polarization(f) for f in frames])
        color = cmap(i % 10)
        plt.plot(steps, va, color=color, label=_eta_legend(eta), linewidth=1.5)

    plt.xlabel("tiempo (steps)")
    plt.ylabel("polarización (va)")
    plt.title("Polarización vs tiempo (comparación)")
    plt.ylim(0.0, 1.05)
    plt.margins(x=0, y=0)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description="Visualizador TP2")
    parser.add_argument("--input", type=Path, default=repo_root / "tp2-output" / "output.xyz")
    parser.add_argument(
        "--gif",
        type=Path,
        default=repo_root / "tp2-visual" / "graphs" / "particles.gif",
        help="GIF de particulas.",
    )
    parser.add_argument(
        "--polarization-gif",
        type=Path,
        default=repo_root / "tp2-visual" / "graphs" / "polarization_time.gif",
        help="GIF de polarización vs tiempo.",
    )
    parser.add_argument(
        "--combined-gif",
        type=Path,
        default=repo_root / "tp2-visual" / "graphs" / "combined.gif",
        help="GIF combinado de particulas y polarización-tiempo.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=repo_root / "tp2-visual" / "graphs" / "polarization_time.png",
        help="PNG de polarización (uno o comparacion segun modo)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PATH", "ETA"),
        action="append",
        default=None,
        help="Trayectoria .xyz y η para la leyenda; repetir para superponer varias curvas",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--no-particles-gif", action="store_true", help="No generar GIF de particulas")
    parser.add_argument(
        "--no-polarization-gif",
        action="store_true",
        help="No generar GIF de polarización-tiempo",
    )
    parser.add_argument("--no-combined-gif", action="store_true", help="No generar GIF combinado")
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Generar solo PNG (desactiva todos los GIFs)",
    )

    args = parser.parse_args()

    if args.compare:
        series: list[tuple[Path, float]] = []
        for path_str, eta_str in args.compare:
            series.append((Path(path_str), float(eta_str)))
        out_png = (
            args.png
            if args.png is not None
            else repo_root / "tp2-visual" / "graphs" / "polarization_compare.png"
        )
        save_polarization_overlay(series, args.stride, args.max_frames, out_png)
        desc = ", ".join(f"{p} (eta={e:g})" for p, e in series)
        print(f"Comparacion ({len(series)} series): {desc}")
        print(f"PNG comparacion generado: {out_png}")
        return

    out_png = args.png
    frames = load_frames(args.input, args.stride, args.max_frames)
    if not frames:
        raise SystemExit(f"No se cargaron frames desde: {args.input}")

    if args.png_only:
        args.no_particles_gif = True
        args.no_polarization_gif = True
        args.no_combined_gif = True

    if not args.no_particles_gif:
        make_particles_animation(frames, args.fps, args.gif, 120)
    if not args.no_polarization_gif:
        make_polarization_animation(frames, args.fps, args.polarization_gif, 120)
    if not args.no_combined_gif:
        make_combined_animation(frames, args.fps, args.combined_gif, 120)
    save_polarization_png(frames, out_png)

    if not args.no_particles_gif:
        print(f"GIF particulas generado: {args.gif}")
    if not args.no_polarization_gif:
        print(f"GIF polarización-tiempo generado: {args.polarization_gif}")
    if not args.no_combined_gif:
        print(f"GIF combinado generado: {args.combined_gif}")
    print(f"PNG polarización-tiempo generado: {out_png}")


if __name__ == "__main__":
    main()
