"""Render a 73-joint sign sequence as a lightweight 2-D skeleton GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 62, 11],
    [0, 1, 4, 7, 59, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
    [21, 37, 38, 39, 69],
    [21, 40, 41, 42, 70],
    [21, 43, 44, 45, 72],
    [21, 46, 47, 48, 71],
    [21, 49, 50, 51, 68],
    [20, 22, 23, 24, 64],
    [20, 25, 26, 27, 65],
    [20, 28, 29, 30, 67],
    [20, 31, 32, 33, 66],
    [20, 34, 35, 36, 63],
    [15, 52],
    [52, 53],
    [52, 54],
    [53, 55],
    [54, 56],
    [62, 60],
    [62, 61],
    [59, 57],
    [59, 58],
]


def load_motion(path: str | Path) -> np.ndarray:
    motion = np.asarray(np.load(path), dtype=np.float32)
    while motion.ndim > 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(
            f"Expected [frames, joints, 3], received {tuple(motion.shape)}"
        )
    required_joints = max(max(chain) for chain in KINEMATIC_CHAINS) + 1
    if motion.shape[1] < required_joints:
        raise ValueError(
            f"The SignGPT skeleton requires at least {required_joints} joints; "
            f"received {motion.shape[1]}."
        )
    return motion


def render_skeleton(
    motion: np.ndarray,
    output: str | Path,
    fps: int = 20,
    plane: str = "xy",
) -> Path:
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in axes:
        raise ValueError(f"Unsupported projection plane: {plane}")

    projected = motion[..., list(axes[plane])]
    mins = projected.min(axis=(0, 1))
    maxs = projected.max(axis=(0, 1))
    center = (mins + maxs) / 2
    span = max(float(np.max(maxs - mins)), 1e-6) * 1.1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    points = ax.scatter([], [], s=12, color="#2563eb")
    lines = LineCollection([], colors="#dc2626", linewidths=1.8)
    ax.add_collection(lines)

    def update(frame_index: int):
        frame = projected[frame_index]
        segments = [
            [frame[start], frame[end]]
            for chain in KINEMATIC_CHAINS
            for start, end in zip(chain[:-1], chain[1:])
        ]
        points.set_offsets(frame)
        lines.set_segments(segments)
        return points, lines

    animation = FuncAnimation(
        fig,
        update,
        frames=len(projected),
        interval=1000 / fps,
        blit=True,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", help="Input .npy file shaped [T, J, 3]")
    parser.add_argument("--output", required=True, help="Output .gif path")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_skeleton(
        load_motion(args.motion),
        output=args.output,
        fps=args.fps,
        plane=args.plane,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
