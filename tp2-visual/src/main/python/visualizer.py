"""
TP2 visualizer: generates a GIF of particle motion with:
- velocity vectors colored by direction angle
- leader highlighted with a fixed color
- polarization curve va(t)

Example:
python3 REPO_NUEVO/tp2-visual/src/main/python/visualizer.py \
  --input REPO_NUEVO/tp2-output/output.xyz \
  --gif REPO_NUEVO/tp2-visual/tp2.gif \
  --polarization-gif REPO_NUEVO/tp2-visual/polarization.gif \
  --stride 8 --max-frames 180
"""

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
    raw = "".join(lines)
    if "," in raw:
        raw = raw.replace(",", ".")
    arr = np.fromstring(raw, sep=" ")
    if arr.size == 0:
        raise ValueError("Empty frame block in output file.")
    if arr.size % n_particles != 0:
        raise ValueError(
            f"Frame parse error: got {arr.size} values for {n_particles} particles."
        )
    cols = arr.size // n_particles
    if cols < 5:
        raise ValueError(f"Expected >= 5 columns per particle, got {cols}.")
    return arr.reshape((n_particles, cols))


def load_frames(path: Path, stride: int, max_frames: int | None) -> list[Frame]:
    frames: list[Frame] = []
    frame_idx = 0

    with path.open("r", encoding="utf-8") as f:
        while True:
            count_line = f.readline()
            if not count_line:
                break
            count_line = count_line.strip()
            if not count_line:
                continue

            try:
                n_particles = int(count_line)
            except ValueError as exc:
                raise ValueError(f"Invalid particle count line: '{count_line}'") from exc

            comment = f.readline()
            if not comment:
                break

            particle_lines = [f.readline() for _ in range(n_particles)]
            if any(line == "" for line in particle_lines):
                break

            if frame_idx % stride == 0:
                data = parse_frame_block(particle_lines, n_particles)
                x = data[:, 1]
                y = data[:, 2]
                vx = data[:, 3]
                vy = data[:, 4]

                if data.shape[1] >= 8:
                    leader_mask = data[:, 7] > 0.5
                else:
                    leader_mask = np.zeros(n_particles, dtype=bool)

                frames.append(
                    Frame(
                        step=parse_step(comment),
                        x=x,
                        y=y,
                        vx=vx,
                        vy=vy,
                        leader_mask=leader_mask,
                    )
                )

                if max_frames is not None and len(frames) >= max_frames:
                    break

            frame_idx += 1

    return frames


def infer_box_size(frames: list[Frame], box_arg: float | None) -> float:
    if box_arg is not None:
        return box_arg
    max_x = max(float(np.max(frame.x)) for frame in frames)
    max_y = max(float(np.max(frame.y)) for frame in frames)
    return max(max_x, max_y) * 1.02


def style_axes(ax: plt.Axes, box_size: float) -> None:
    ax.set_xlim(0.0, box_size)
    ax.set_ylim(0.0, box_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    ax.grid(color="#e1e1e1", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for spine in ax.spines.values():
        spine.set_color("#5a5a5a")
        spine.set_linewidth(1.0)


def compute_polarization(frame: Frame, include_leader: bool) -> float:
    # TP2/Theory: va = |sum(v_i)| / (N*v). Here vx,vy are unit directions, so va = |sum(n_i)| / N.
    if include_leader:
        selected = np.ones(frame.vx.shape[0], dtype=bool)
    else:
        selected = ~frame.leader_mask

    if not np.any(selected):
        return 0.0

    sum_vx = float(np.sum(frame.vx[selected]))
    sum_vy = float(np.sum(frame.vy[selected]))
    return math.hypot(sum_vx, sum_vy) / float(np.sum(selected))


def make_animation(
    frames: list[Frame],
    box_size: float,
    vector_scale: float,
    fps: int,
    gif_path: Path,
    png_path: Path | None,
    dpi: int,
    leader_color: str,
    leader_size: float,
    show_leader_trail: bool,
    trail_length: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    style_axes(ax, box_size)

    norm = plt.Normalize(0.0, 2.0 * math.pi)
    cmap = plt.get_cmap("hsv")

    first = frames[0]
    angle0 = np.mod(np.arctan2(first.vy, first.vx), 2.0 * math.pi)

    non_leader0 = ~first.leader_mask
    q_non = ax.quiver(
        first.x[non_leader0],
        first.y[non_leader0],
        first.vx[non_leader0] * vector_scale,
        first.vy[non_leader0] * vector_scale,
        angle0[non_leader0],
        cmap=cmap,
        norm=norm,
        scale_units="xy",
        scale=1.0,
        width=0.0042,
        headwidth=3.8,
        headlength=5.0,
        headaxislength=4.5,
        pivot="mid",
        alpha=0.96,
        zorder=2,
    )

    q_leader = ax.quiver(
        first.x[first.leader_mask],
        first.y[first.leader_mask],
        first.vx[first.leader_mask] * vector_scale,
        first.vy[first.leader_mask] * vector_scale,
        color=leader_color,
        scale_units="xy",
        scale=1.0,
        width=0.0068,
        headwidth=4.8,
        headlength=6.0,
        headaxislength=5.2,
        pivot="mid",
        alpha=0.98,
        zorder=3,
    )

    leader_pts = ax.scatter(
        first.x[first.leader_mask],
        first.y[first.leader_mask],
        s=leader_size,
        c=leader_color,
        edgecolors="#1b1b1b",
        linewidths=0.7,
        marker="o",
        zorder=4,
        label="Leader",
    )

    has_leader = any(np.any(frame.leader_mask) for frame in frames)
    if has_leader:
        ax.legend(loc="upper right", frameon=True, edgecolor="#666", fontsize=9)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.05,
        pad=0.02,
    )
    cbar.set_label("Velocity angle (rad)")

    title = ax.set_title(
        "TP2 - Vicsek with leader | colored by velocity angle",
        fontsize=12,
    )
    step_text = ax.text(
        0.02,
        0.98,
        f"step = {first.step}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#111",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
    )

    leader_trail_x: list[float] = []
    leader_trail_y: list[float] = []
    (trail_line,) = ax.plot([], [], color=leader_color, linewidth=1.0, alpha=0.7, zorder=1)

    def update(i: int):
        frame = frames[i]
        angles = np.mod(np.arctan2(frame.vy, frame.vx), 2.0 * math.pi)

        non_leader = ~frame.leader_mask
        q_non.set_offsets(np.column_stack((frame.x[non_leader], frame.y[non_leader])))
        q_non.set_UVC(
            frame.vx[non_leader] * vector_scale,
            frame.vy[non_leader] * vector_scale,
            angles[non_leader],
        )

        q_leader.set_offsets(np.column_stack((frame.x[frame.leader_mask], frame.y[frame.leader_mask])))
        q_leader.set_UVC(
            frame.vx[frame.leader_mask] * vector_scale,
            frame.vy[frame.leader_mask] * vector_scale,
        )
        leader_pts.set_offsets(np.column_stack((frame.x[frame.leader_mask], frame.y[frame.leader_mask])))

        if show_leader_trail and np.any(frame.leader_mask):
            x_lead = float(frame.x[frame.leader_mask][0])
            y_lead = float(frame.y[frame.leader_mask][0])
            leader_trail_x.append(x_lead)
            leader_trail_y.append(y_lead)
            if len(leader_trail_x) > trail_length:
                leader_trail_x.pop(0)
                leader_trail_y.pop(0)
            trail_line.set_data(leader_trail_x, leader_trail_y)
        else:
            trail_line.set_data([], [])

        title.set_text("TP2 - Vicsek with leader | colored by velocity angle")
        step_text.set_text(f"step = {frame.step}")

        return q_non, q_leader, leader_pts, trail_line, title, step_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )

    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            "Could not save GIF with PillowWriter. Install Pillow "
            "(python3 -m pip install pillow)."
        ) from exc
    finally:
        plt.close(fig)


def make_polarization_animation(
    frames: list[Frame],
    fps: int,
    gif_path: Path,
    dpi: int,
    include_leader: bool,
    x_max: float | None,
    final_png_path: Path | None,
) -> None:
    steps = np.array([frame.step for frame in frames], dtype=float)
    va = np.array([compute_polarization(frame, include_leader) for frame in frames], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")
    ax.grid(color="#e1e1e1", linewidth=0.7, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#5a5a5a")
        spine.set_linewidth(1.0)

    if len(steps) == 1:
        x_min, x_max_data = steps[0] - 1.0, steps[0] + 1.0
    else:
        x_min, x_max_data = float(np.min(steps)), float(np.max(steps))

    x_upper = x_max_data if x_max is None else float(x_max)
    if x_upper <= x_min:
        x_upper = x_max_data

    ax.set_xlim(x_min, x_upper)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("time step")
    ax.set_ylabel("polarization va")
    ax.set_title("TP2 - Polarization vs time")

    line, = ax.plot([], [], color="#0a58ca", linewidth=2.2)
    point = ax.scatter([], [], s=44, c="#d62828", edgecolors="#222222", linewidths=0.5, zorder=3)
    label = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f6f6f6", "edgecolor": "#bbbbbb"},
    )

    def update(i: int):
        line.set_data(steps[: i + 1], va[: i + 1])
        point.set_offsets(np.array([[steps[i], va[i]]]))
        label.set_text(f"step = {int(steps[i])}\nva = {va[i]:.4f}")
        return line, point, label

    if final_png_path is not None:
        final_png_path.parent.mkdir(parents=True, exist_ok=True)
        ax.plot(steps, va, color="#0a58ca", linewidth=2.2)
        ax.scatter(
            [steps[-1]],
            [va[-1]],
            s=58,
            c="#d62828",
            edgecolors="#222222",
            linewidths=0.6,
            zorder=4,
        )
        ax.text(
            0.98,
            0.03,
            f"final step = {int(steps[-1])}\nva = {va[-1]:.4f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=10,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f6f6f6", "edgecolor": "#bbbbbb"},
        )
        fig.savefig(final_png_path, dpi=dpi)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            "Could not save polarization GIF with PillowWriter. Install Pillow "
            "(python3 -m pip install pillow)."
        ) from exc
    finally:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description="Generate TP2 GIF: vectors colored by angle and leader highlighted."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "tp2-output" / "output.xyz",
        help="Path to output.xyz generated by the simulator.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=repo_root / "tp2-visual" / "tp2.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--polarization-gif",
        type=Path,
        default=repo_root / "tp2-visual" / "polarization.gif",
        help="Output polarization-vs-time GIF path.",
    )
    parser.add_argument(
        "--no-polarization-gif",
        action="store_true",
        help="Do not generate the polarization GIF.",
    )
    parser.add_argument(
        "--polarization-final-png",
        type=Path,
        default=repo_root / "tp2-visual" / "polarization_final.png",
        help="Output static PNG with the final polarization curve.",
    )
    parser.add_argument(
        "--no-spatial-gif",
        action="store_true",
        help="Do not generate the particle-motion GIF.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional still image path (first frame).",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        help="Simulation box side length L. If omitted, inferred from data.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Use one frame every N from output.xyz (default: 8).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Maximum number of frames in GIF (default: 180).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=0.35,
        help="Scale factor applied to velocity vectors.",
    )
    parser.add_argument(
        "--leader-color",
        type=str,
        default="#d62828",
        help="Leader color.",
    )
    parser.add_argument(
        "--leader-size",
        type=float,
        default=62.0,
        help="Leader marker size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output DPI for GIF/PNG.",
    )
    parser.add_argument(
        "--show-leader-trail",
        action="store_true",
        help="Draw a short trail for the leader.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Trail length in frames when --show-leader-trail is enabled.",
    )
    parser.add_argument(
        "--exclude-leader-from-va",
        action="store_true",
        help="Compute va(t) excluding leader particles.",
    )
    parser.add_argument(
        "--polarization-xmax",
        type=float,
        default=None,
        help="Force x-axis upper limit for polarization GIF (e.g., 4000).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.stride <= 0:
        raise SystemExit("--stride must be > 0.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be > 0 when provided.")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.vector_scale <= 0:
        raise SystemExit("--vector-scale must be > 0.")
    if args.trail_length <= 0:
        raise SystemExit("--trail-length must be > 0.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    frames = load_frames(args.input, args.stride, args.max_frames)
    if not frames:
        raise SystemExit("No frames were loaded from output.xyz.")

    if not args.no_spatial_gif:
        box_size = infer_box_size(frames, args.box)
        make_animation(
            frames=frames,
            box_size=box_size,
            vector_scale=args.vector_scale,
            fps=args.fps,
            gif_path=args.gif,
            png_path=args.png,
            dpi=args.dpi,
            leader_color=args.leader_color,
            leader_size=args.leader_size,
            show_leader_trail=args.show_leader_trail,
            trail_length=args.trail_length,
        )
    if not args.no_polarization_gif:
        make_polarization_animation(
            frames=frames,
            fps=args.fps,
            gif_path=args.polarization_gif,
            dpi=args.dpi,
            include_leader=not args.exclude_leader_from_va,
            x_max=args.polarization_xmax,
            final_png_path=args.polarization_final_png,
        )

    if not args.no_spatial_gif:
        print(f"GIF generated: {args.gif}")
    if not args.no_polarization_gif:
        print(f"Polarization GIF generated: {args.polarization_gif}")
        print(f"Polarization final PNG generated: {args.polarization_final_png}")
    if args.png is not None and not args.no_spatial_gif:
        print(f"PNG generated: {args.png}")


if __name__ == "__main__":
    main()
