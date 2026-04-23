from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "Matplotlib no esta instalado. Ejecuta 'python3 -m pip install matplotlib numpy pillow' y reintenta."
    )

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class StablePaletteGifWriter(animation.PillowWriter):
    """PillowWriter con paleta global compartida entre todos los frames.

    Pillow asigna por defecto una paleta distinta a cada frame del GIF, lo que
    causa que colores que deberían ser fijos (la leyenda, el color propio de
    cada cliente durante su estadía) se muevan levemente frame a frame. Acá
    construimos una sola paleta a partir de una muestra uniforme de frames y
    la aplicamos a todos, eliminando ese flicker.
    """

    def finish(self) -> None:
        from PIL import Image

        if not self._frames:
            return
        frames = self._frames
        width, height = frames[0].size
        n_samples = min(len(frames), 48)
        step = max(1, len(frames) // n_samples)
        indices = list(range(0, len(frames), step))[:n_samples]
        if indices[-1] != len(frames) - 1:
            indices.append(len(frames) - 1)

        montage = Image.new("RGB", (width, height * len(indices)))
        for row, idx in enumerate(indices):
            montage.paste(frames[idx].convert("RGB"), (0, row * height))
        palette_img = montage.quantize(colors=255, method=0, dither=0)

        quantized = [
            f.convert("RGB").quantize(palette=palette_img, dither=0)
            for f in frames
        ]
        quantized[0].save(
            self.outfile,
            save_all=True,
            append_images=quantized[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2,
            optimize=False,
        )


def _build_color_pool(size: int = 96, offset: int = 0) -> list[tuple[float, float, float, float]]:
    """Paleta bien distribuida en TODO el círculo cromático.

    Se recorre el hue con el golden ratio (máxima dispersión) y se varía
    sutilmente saturación y brillo para obtener muchos hexadecimales distintos
    sin quedar concentrados en un rango de colores. El pool puede ampliarse
    pasando un `offset` para que los colores nuevos no repitan los anteriores.
    """
    golden = 0.618033988749895
    sat_steps = (0.95, 0.75, 0.60)
    val_steps = (0.95, 0.80, 0.65)
    pool: list[tuple[float, float, float, float]] = []
    for i in range(size):
        idx = offset + i
        hue = (idx * golden) % 1.0
        sat = sat_steps[idx % len(sat_steps)]
        val = val_steps[(idx // len(sat_steps)) % len(val_steps)]
        r, g, b = mcolors.hsv_to_rgb((hue, sat, val))
        pool.append((float(r), float(g), float(b), 1.0))
    return pool


SERVER_COLORS = {
    "FREE": "#80ed99",
    "BUSY": "#ff595e",
}

CLIENT_EDGE_COLOR = "#222222"


@dataclass
class ClientSnapshot:
    client_id: int
    x: float
    y: float
    state: str
    assigned_server_id: int
    queue_spot_index: int
    rgb: tuple[int, int, int]


@dataclass
class ServerSnapshot:
    server_id: int
    x: float
    y: float
    status: str
    current_client_id: int


@dataclass
class Frame:
    time: float
    clients: list[ClientSnapshot]
    servers: list[ServerSnapshot]


@dataclass
class SimulationData:
    frames: list[Frame]
    metadata: dict[str, str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_input_path() -> Path:
    output_dir = repo_root() / "tp3-output"
    candidates = sorted(output_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(
            f"No hay archivos .txt en '{output_dir}'. Genera una simulacion de TP3 antes de visualizarla."
        )
    return candidates[0]


def default_output_path(input_path: Path) -> Path:
    return repo_root() / "tp3-visual" / "graphs" / f"{input_path.stem}.gif"


def parse_key_value_line(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def load_frames(path: Path, stride: int, max_frames: int | None) -> SimulationData:
    frames: list[Frame] = []
    metadata: dict[str, str] = {}

    current_time: float | None = None
    current_clients: list[ClientSnapshot] = []
    current_servers: list[ServerSnapshot] = []
    frame_idx = 0
    in_stats = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line == "STATS":
                in_stats = True
                continue

            if in_stats:
                metadata.update(parse_key_value_line(line))
                continue

            if line.startswith("TIME "):
                current_time = float(line.split()[1])
                current_clients = []
                current_servers = []
                continue

            if line == "---":
                if current_time is None:
                    continue
                if frame_idx % stride == 0:
                    frames.append(Frame(current_time, current_clients, current_servers))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                frame_idx += 1
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "CLIENT" and len(parts) == 10:
                current_clients.append(
                    ClientSnapshot(
                        client_id=int(parts[1]),
                        x=float(parts[2]),
                        y=float(parts[3]),
                        state=parts[4],
                        assigned_server_id=int(parts[5]),
                        queue_spot_index=int(parts[6]),
                        rgb=(int(parts[7]), int(parts[8]), int(parts[9])),
                    )
                )
                continue

            if parts[0] == "SERVER" and len(parts) == 6:
                current_servers.append(
                    ServerSnapshot(
                        server_id=int(parts[1]),
                        x=float(parts[2]),
                        y=float(parts[3]),
                        status=parts[4],
                        current_client_id=int(parts[5]),
                    )
                )

    if not frames:
        raise ValueError(f"No se pudieron cargar frames desde '{path}'.")

    return SimulationData(frames=frames, metadata=metadata)


def client_sizes(clients: list[ClientSnapshot]) -> np.ndarray:
    if not clients:
        return np.empty((0,))
    return np.array([
        120.0 if client.state == "BEING_SERVED" else 95.0
        for client in clients
    ])


def server_facecolors(servers: list[ServerSnapshot]) -> np.ndarray:
    return np.array([
        mcolors.to_rgba(SERVER_COLORS.get(server.status, "#adb5bd"), alpha=1.0)
        for server in servers
    ])


def waiting_count(frame: Frame) -> int:
    waiting_states = {"WALKING_TO_QUEUE_SPOT", "IN_QUEUE", "ADVANCING_IN_QUEUE"}
    return sum(client.state in waiting_states for client in frame.clients)


def walking_to_server_count(frame: Frame) -> int:
    return sum(client.state == "WALKING_TO_SERVER" for client in frame.clients)


def being_served_count(frame: Frame) -> int:
    return sum(client.state == "BEING_SERVED" for client in frame.clients)


def build_header(input_path: Path, metadata: dict[str, str]) -> str:
    if not metadata:
        return input_path.stem

    pieces = []
    queue_type = metadata.get("queueType")
    if queue_type is not None:
        pieces.append(f"tipo de fila={queue_type}")

    for key in ("t1", "t2"):
        value = metadata.get(key)
        if value is not None:
            pieces.append(f"{key}={value} s")

    k_value = metadata.get("k")
    if k_value is not None:
        pieces.append(f"k={k_value}")

    return " | ".join(pieces) if pieces else input_path.stem


def render_gif(
    input_path: Path,
    output_path: Path,
    room_size: float,
    fps: int,
    dpi: int,
    stride: int,
    max_frames: int | None,
) -> None:
    data = load_frames(input_path, stride=stride, max_frames=max_frames)
    frames = data.frames

    first_frame = frames[0]
    server_positions = np.array([[server.x, server.y] for server in first_frame.servers], dtype=float)

    fig, ax_room = plt.subplots(figsize=(9.0, 10.0))
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.10, right=0.96)

    fig.text(
        0.5, 0.02,
        build_header(input_path, data.metadata),
        ha="center", va="bottom",
        fontsize=11,
        color="#495057",
    )

    ax_room.set_xlabel("x (m)")
    ax_room.set_ylabel("y (m)")
    ax_room.set_xlim(0.0, room_size)
    ax_room.set_ylim(-1.5, room_size)
    ax_room.set_aspect("equal")
    ax_room.grid(True, alpha=0.15)

    # Invisible vertical strip boundaries, one per server
    num_servers = len(first_frame.servers)
    if num_servers > 1:
        strip_width = room_size / num_servers
        for i in range(1, num_servers):
            ax_room.axvline(
                x=strip_width * i,
                color="#adb5bd",
                linewidth=1.0,
                linestyle="--",
                alpha=0.45,
                zorder=1,
            )

    room_info = ax_room.text(
        0.01,
        0.99,
        "",
        transform=ax_room.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8f9fa", "edgecolor": "#ced4da"},
    )

    client_scatter = ax_room.scatter(
        [],
        [],
        s=np.empty((0,)),
        facecolors=np.empty((0, 4)),
        edgecolors=CLIENT_EDGE_COLOR,
        linewidths=0.8,
        zorder=3,
    )
    # Aseguramos que no quede activo el modo "colormap" del scatter: si queda
    # activo, los set_facecolors posteriores pueden ser sobreescritos al dibujar
    # y las partículas aparecerían sin color.
    client_scatter.set_array(None)

    server_scatter = ax_room.scatter(
        server_positions[:, 0] if len(server_positions) else [],
        server_positions[:, 1] if len(server_positions) else [],
        s=700,
        marker="s",
        c=server_facecolors(first_frame.servers),
        edgecolors="#343a40",
        linewidths=1.2,
        zorder=2,
    )

    server_labels = []
    for server in first_frame.servers:
        label = ax_room.text(
            server.x,
            server.y + 0.9,
            "",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#212529",
            zorder=4,
        )
        server_labels.append(label)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=SERVER_COLORS["FREE"],
               markeredgecolor="#343a40", markersize=9, linewidth=0, label="Servidor libre"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=SERVER_COLORS["BUSY"],
               markeredgecolor="#343a40", markersize=9, linewidth=0, label="Servidor ocupado"),
    ]
    ax_room.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.95)

    def update(frame_index: int):
        frame = frames[frame_index]
        ordered_clients = sorted(
            frame.clients,
            key=lambda client: client.state == "BEING_SERVED",
        )

        if ordered_clients:
            client_offsets = np.array([[client.x, client.y] for client in ordered_clients], dtype=float)
        else:
            client_offsets = np.empty((0, 2))

        face_colors = np.array([
            (client.rgb[0] / 255.0, client.rgb[1] / 255.0, client.rgb[2] / 255.0, 1.0)
            for client in ordered_clients
        ]) if ordered_clients else np.empty((0, 4))

        client_scatter.set_offsets(client_offsets)
        client_scatter.set_facecolors(face_colors)
        client_scatter.set_sizes(client_sizes(ordered_clients))

        current_server_positions = np.array([[server.x, server.y] for server in frame.servers], dtype=float)
        if len(current_server_positions) == 0:
            current_server_positions = np.empty((0, 2))
        server_scatter.set_offsets(current_server_positions)
        server_scatter.set_facecolors(server_facecolors(frame.servers))

        for label, server in zip(server_labels, frame.servers):
            if server.status == "BUSY" and server.current_client_id >= 0:
                label.set_text(f"S{server.server_id}")
            else:
                label.set_text(f"S{server.server_id}")

        room_info.set_text(
            f"t = {frame.time:.2f} s\n"
            f"clientes totales = {len(frame.clients)}\n"
            f"en cola = {waiting_count(frame)}\n"
            f"caminando al servidor = {walking_to_server_count(frame)}\n"
            f"siendo atendidos = {being_served_count(frame)}"
        )

        artists = [
            client_scatter,
            server_scatter,
            room_info,
        ]
        artists.extend(server_labels)
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = len(frames)
    print(f"Generando GIF: {total_frames} frames  dpi={dpi}  fps={fps}  → {output_path}")

    def _progress(current: int, total: int) -> None:
        pct = int(100 * current / total) if total else 100
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  frame {current}/{total}", end="", flush=True)

    anim.save(
        output_path,
        writer=StablePaletteGifWriter(fps=fps),
        dpi=dpi,
        progress_callback=_progress,
    )
    print()  # salto de línea tras la barra de progreso
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualizador TP3: genera un GIF a partir de un output del simulador.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Archivo .txt del TP3 a visualizar. Si no se indica, toma el mas reciente de tp3-output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del GIF de salida. Default: tp3-visual/graphs/<nombre_input>.gif",
    )
    parser.add_argument("--room-size", type=float, default=30.0, help="Lado del recinto en metros. Default: 30")
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo del GIF. Default: 10")
    parser.add_argument("--dpi", type=int, default=80, help="Resolucion del GIF. Default: 80")
    parser.add_argument("--stride", type=int, default=2, help="Tomar 1 de cada N frames del output. Default: 2")
    parser.add_argument("--max-frames", type=int, default=None, help="Limitar cantidad maxima de frames")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve() if args.input is not None else default_input_path()
    if not input_path.exists():
        sys.exit(f"No existe el archivo de input '{input_path}'.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    if args.stride <= 0:
        sys.exit("--stride debe ser mayor a 0")
    if args.fps <= 0:
        sys.exit("--fps debe ser mayor a 0")
    if args.dpi <= 0:
        sys.exit("--dpi debe ser mayor a 0")
    if args.room_size <= 0:
        sys.exit("--room-size debe ser mayor a 0")

    render_gif(
        input_path=input_path,
        output_path=output_path,
        room_size=args.room_size,
        fps=args.fps,
        dpi=args.dpi,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    print(f"GIF generado en: {output_path}")


if __name__ == "__main__":
    main()
