# VISUALIZADOR TP2
# - GIF de particulas (particles.gif)
# - GIF y PNG de polarizacion vs tiempo (polarization_time.gif/.png)
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


DEFAULT_STATIONARY_STEP = 0.0
STATIONARY_COLOR = "#d62828"


@dataclass
class Frame:
    step: int
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    leader_mask: np.ndarray


@dataclass
class PolarizationSeries:
    eta: float
    label: str
    steps: np.ndarray
    va: np.ndarray
    stationary_confirm_idx: int | None
    stationary_va: float | None


@dataclass
class CompareSpec:
    path: Path
    eta: float


def parse_step(comment_line: str) -> int:
    match = re.search(r"-?\d+", comment_line)
    return int(match.group(0)) if match else 0


def parse_frame_block(lines: list[str], n_particles: int) -> np.ndarray:
    data = []
    for line in lines:
        parts = line.strip().split()
        # Formatos soportados:
        # - id x y vx vy leader
        # - id x y vx vy ... (columnas extra legacy)
        # En formato legacy tomamos leader=0 para no romper el parseo.
        if len(parts) < 5:
            continue
        try:
            pid = float(parts[0])
            px = float(parts[1])
            py = float(parts[2])
            pvx = float(parts[3])
            pvy = float(parts[4])
            if len(parts) == 6:
                leader = float(parts[5])
            else:
                leader = 0.0
            row = [pid, px, py, pvx, pvy, leader]
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


def stationary_confirm_index(steps: np.ndarray, stationary_step: float) -> int | None:
    if steps.size == 0:
        return None

    idx = int(np.searchsorted(steps, stationary_step, side="left"))
    if idx >= steps.size:
        return None
    return idx


def mark_stationary_step_on_xaxis(ax: plt.Axes, stationary_step: float) -> None:
    current_ticks = np.asarray(ax.get_xticks(), dtype=float)
    if current_ticks.size == 0:
        ticks = np.array([stationary_step], dtype=float)
    else:
        ticks = np.sort(np.unique(np.append(current_ticks, stationary_step)))

    ax.set_xticks(ticks)

    labels: list[str] = []
    for tick in ticks:
        if float(tick).is_integer():
            labels.append(str(int(round(tick))))
        else:
            labels.append(f"{tick:g}")
    ax.set_xticklabels(labels)

    for tick, tick_label in zip(ticks, ax.get_xticklabels()):
        if np.isclose(tick, stationary_step, atol=1e-12, rtol=0.0):
            tick_label.set_color(STATIONARY_COLOR)
            tick_label.set_fontweight("bold")


def build_polarization_series(
    frames: list[Frame],
    eta: float,
    label: str,
    stationary_step: float,
) -> PolarizationSeries:
    steps = np.array([f.step for f in frames], dtype=float)
    va = np.array([compute_polarization(f) for f in frames], dtype=float)
    confirm_idx = stationary_confirm_index(steps, stationary_step)
    stationary_va = float(va[confirm_idx]) if confirm_idx is not None else None
    return PolarizationSeries(
        eta=eta,
        label=label,
        steps=steps,
        va=va,
        stationary_confirm_idx=confirm_idx,
        stationary_va=stationary_va,
    )


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


def make_combined_animation(
    frames: list[Frame],
    fps: int,
    gif_path: Path,
    dpi: int,
    stationary_step: float,
):
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
    if len(steps) == 1:
        ax_va.set_xlim(float(steps[0]) - 1.0, float(steps[0]) + 1.0)
    else:
        ax_va.set_xlim(float(steps.min()), float(steps.max()))
    ax_va.set_ylim(0.0, 1.05)
    ax_va.grid(True, alpha=0.3)

    stationary_confirm_idx = stationary_confirm_index(steps, stationary_step)
    stationary_line = None
    x_min, x_max = ax_va.get_xlim()
    if stationary_confirm_idx is not None and x_min <= stationary_step <= x_max:
        mark_stationary_step_on_xaxis(ax_va, stationary_step)
        stationary_line = ax_va.axvline(
            stationary_step,
            color=STATIONARY_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
        stationary_line.set_visible(False)

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

        if stationary_line is not None and stationary_confirm_idx is not None:
            is_visible = i >= stationary_confirm_idx
            stationary_line.set_visible(is_visible)

        artists = [q_no_lider, step_text, line, point, label]
        if q_lider is not None and puntos_lider is not None:
            artists.extend([q_lider, puntos_lider])
        if stationary_line is not None:
            artists.append(stationary_line)
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def make_polarization_animation(
    frames: list[Frame],
    fps: int,
    gif_path: Path,
    dpi: int,
    stationary_step: float,
):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.87, bottom=0.25)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarización (va)")

    if len(steps) == 1:
        ax.set_xlim(float(steps[0]) - 1.0, float(steps[0]) + 1.0)
    else:
        ax.set_xlim(float(steps.min()), float(steps.max()))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    stationary_confirm_idx = stationary_confirm_index(steps, stationary_step)
    stationary_line = None
    x_min, x_max = ax.get_xlim()
    if stationary_confirm_idx is not None and x_min <= stationary_step <= x_max:
        mark_stationary_step_on_xaxis(ax, stationary_step)
        stationary_line = ax.axvline(
            stationary_step,
            color=STATIONARY_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
        stationary_line.set_visible(False)

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
        if stationary_line is not None and stationary_confirm_idx is not None:
            is_visible = i >= stationary_confirm_idx
            stationary_line.set_visible(is_visible)
            artists.append(stationary_line)

        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_polarization_png(frames: list[Frame], path: Path, stationary_step: float):
    steps = np.array([f.step for f in frames])
    va = np.array([compute_polarization(f) for f in frames])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25)
    ax.plot(steps, va, color="#0a58ca", linewidth=2.0)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarización (va)")
    ax.set_xlim(float(np.min(steps)), float(np.max(steps)))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0, y=0)

    stationary_confirm_idx = stationary_confirm_index(steps, stationary_step)
    if stationary_confirm_idx is not None:
        x_min, x_max = ax.get_xlim()
        if x_min <= stationary_step <= x_max:
            mark_stationary_step_on_xaxis(ax, stationary_step)
            ax.axvline(
                stationary_step,
                color=STATIONARY_COLOR,
                linewidth=2.0,
                linestyle="--",
                alpha=0.95,
                zorder=4,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _eta_legend(eta: float) -> str:
    return f"η = {eta:g}"


def load_polarization_series(
    series: list[CompareSpec],
    stride: int,
    max_frames: int | None,
    stationary_step: float,
) -> list[PolarizationSeries]:
    loaded: list[PolarizationSeries] = []
    for spec in series:
        frames = load_frames(spec.path, stride, max_frames)
        if not frames:
            continue
        loaded.append(
            build_polarization_series(
                frames,
                spec.eta,
                _eta_legend(spec.eta),
                stationary_step,
            )
        )
    return loaded


def make_polarization_overlay_animation(
    series_data: list[PolarizationSeries],
    fps: int,
    gif_path: Path,
    dpi: int,
    stationary_step: float,
) -> None:
    if not series_data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarizacion (va)")

    x_min = min(float(np.min(series.steps)) for series in series_data)
    x_max = max(float(np.max(series.steps)) for series in series_data)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    global_stationary_line = None
    has_stationary = x_min <= stationary_step <= x_max
    stationary_confirm_indices: list[int] = []
    if has_stationary:
        mark_stationary_step_on_xaxis(ax, stationary_step)
        global_stationary_line = ax.axvline(
            stationary_step,
            color=STATIONARY_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
        global_stationary_line.set_visible(False)

    for series in series_data:
        if series.stationary_confirm_idx is not None:
            stationary_confirm_indices.append(int(series.stationary_confirm_idx))

    cmap = plt.get_cmap("tab10")
    lines: list = []
    points: list = []
    for i, series in enumerate(series_data):
        color = cmap(i % 10)
        line, = ax.plot([], [], color=color, linewidth=2.0, label=series.label)
        point = ax.scatter([], [], s=38, c=[color], edgecolors="#1b1b1b", linewidths=0.4, zorder=4)
        lines.append(line)
        points.append(point)

    if len(series_data) > 1:
        ax.legend(loc="upper right", fontsize=9)

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

    max_len = max(int(series.steps.size) for series in series_data)

    def update(frame_idx: int):
        artists: list = [label]
        if global_stationary_line is not None:
            artists.append(global_stationary_line)

        any_stationary_visible = any(frame_idx >= idx for idx in stationary_confirm_indices)

        label_lines: list[str] = []

        for i, series in enumerate(series_data):
            if series.steps.size == 0:
                continue

            end_idx = min(frame_idx, int(series.steps.size) - 1)
            lines[i].set_data(series.steps[: end_idx + 1], series.va[: end_idx + 1])
            points[i].set_offsets(np.array([[series.steps[end_idx], series.va[end_idx]]]))
            label_lines.append(
                f"{series.label}: step={int(series.steps[end_idx])} va={series.va[end_idx]:.4f}"
            )
            artists.extend([lines[i], points[i]])

        if global_stationary_line is not None:
            global_stationary_line.set_visible(any_stationary_visible)

        label.set_text("\n".join(label_lines))
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=max_len, interval=1000 / fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_polarization_overlay_series(
    series_data: list[PolarizationSeries],
    path: Path,
    stationary_step: float,
) -> None:
    if not series_data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25)

    cmap = plt.get_cmap("tab10")
    x_min = min(float(np.min(series.steps)) for series in series_data)
    x_max = max(float(np.max(series.steps)) for series in series_data)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarizacion (va)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.margins(x=0, y=0)
    ax.grid(True, alpha=0.3)

    has_stationary = x_min <= stationary_step <= x_max

    for i, series in enumerate(series_data):
        color = cmap(i % 10)
        ax.plot(series.steps, series.va, color=color, linewidth=2.0, label=series.label)

    if has_stationary:
        mark_stationary_step_on_xaxis(ax, stationary_step)
        ax.axvline(
            stationary_step,
            color=STATIONARY_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )

    if len(series_data) > 1:
        ax.legend(loc="upper right", fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _individual_series_png_path(base_path: Path, series: PolarizationSeries) -> Path:
    noise_value = f"{series.eta:g}"
    filename = f"compare_series_{noise_value}.png"
    return base_path.with_name(filename)


def save_single_series_png(series: PolarizationSeries, path: Path, stationary_step: float) -> None:
    if series.steps.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25)
    ax.plot(series.steps, series.va, color="#0a58ca", linewidth=2.0)
    ax.set_xlabel("tiempo (steps)")
    ax.set_ylabel("polarizacion (va)")
    ax.set_title(series.label)

    x_min = float(np.min(series.steps))
    x_max = float(np.max(series.steps))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0, y=0)

    if (
        series.stationary_confirm_idx is not None
        and x_min <= stationary_step <= x_max
    ):
        mark_stationary_step_on_xaxis(ax, stationary_step)
        ax.axvline(
            stationary_step,
            color=STATIONARY_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_individual_series_pngs(
    series_data: list[PolarizationSeries],
    base_path: Path,
    stationary_step: float,
) -> list[Path]:
    outputs: list[Path] = []
    for series in series_data:
        output_path = _individual_series_png_path(base_path, series)
        save_single_series_png(series, output_path, stationary_step)
        outputs.append(output_path)
    return outputs


def save_polarization_overlay(
    series: list[tuple[Path, float] | CompareSpec],
    stride: int,
    max_frames: int | None,
    path: Path,
    stationary_step: float = DEFAULT_STATIONARY_STEP,
) -> None:
    normalized: list[CompareSpec] = []
    for item in series:
        if isinstance(item, CompareSpec):
            normalized.append(item)
        else:
            xyz, eta = item
            normalized.append(CompareSpec(xyz, eta))

    loaded_series = load_polarization_series(normalized, stride, max_frames, stationary_step)
    save_polarization_overlay_series(loaded_series, path, stationary_step)


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
        nargs="+",
        metavar="VALUE",
        action="append",
        default=None,
        help="Serie en formato: PATH ETA. Repetir para superponer varias curvas.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--stationary-step",
        type=float,
        default=DEFAULT_STATIONARY_STEP,
        help="Step fijo desde el cual marcar estacionario para todas las curvas.",
    )
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

    if args.stationary_step < 0:
        raise SystemExit("--stationary-step debe ser >= 0")

    if args.compare:
        if args.png_only:
            args.no_polarization_gif = True

        series: list[CompareSpec] = []
        for compare_entry in args.compare:
            if len(compare_entry) != 2:
                raise SystemExit("Cada --compare debe tener: PATH ETA")

            path_str = compare_entry[0]
            eta_str = compare_entry[1]

            series.append(CompareSpec(Path(path_str), float(eta_str)))

        loaded_series = load_polarization_series(
            series,
            args.stride,
            args.max_frames,
            args.stationary_step,
        )
        if not loaded_series:
            raise SystemExit("No se cargaron corridas validas en --compare")

        out_png = args.png
        if not args.no_polarization_gif:
            make_polarization_overlay_animation(
                loaded_series,
                args.fps,
                args.polarization_gif,
                120,
                args.stationary_step,
            )
        save_polarization_overlay_series(loaded_series, out_png, args.stationary_step)
        individual_pngs = save_individual_series_pngs(
            loaded_series,
            out_png,
            args.stationary_step,
        )

        desc_items: list[str] = []
        for spec in series:
            desc_items.append(f"{spec.path} (eta={spec.eta:g})")
        desc = ", ".join(desc_items)
        print(f"Comparacion ({len(series)} series): {desc}")
        if not args.no_polarization_gif:
            print(f"GIF polarizacion-tiempo generado: {args.polarization_gif}")
        print(f"PNG polarizacion-tiempo generado: {out_png}")
        for png_path in individual_pngs:
            print(f"PNG individual generado: {png_path}")
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
        make_polarization_animation(frames, args.fps, args.polarization_gif, 120, args.stationary_step)
    if not args.no_combined_gif:
        make_combined_animation(frames, args.fps, args.combined_gif, 120, args.stationary_step)
    save_polarization_png(frames, out_png, args.stationary_step)

    if not args.no_particles_gif:
        print(f"GIF particulas generado: {args.gif}")
    if not args.no_polarization_gif:
        print(f"GIF polarización-tiempo generado: {args.polarization_gif}")
    if not args.no_combined_gif:
        print(f"GIF combinado generado: {args.combined_gif}")
    print(f"PNG polarización-tiempo generado: {out_png}")


if __name__ == "__main__":
    main()
