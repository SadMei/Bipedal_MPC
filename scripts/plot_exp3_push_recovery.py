#!/usr/bin/env python3
"""Plot the phase-aligned push-recovery experiment in paper style."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle


INK = "#20242C"
MUTED = "#687080"
GRID = "#E2E6ED"
AXIS = "#C9CFD9"
SRBM = "#6485C7"
IR = "#D9473F"
IRM = "#8D2F7C"
SRBM_FILL = "#BDD0EE"
IR_FILL = "#F2AAA5"
IRM_FILL = "#D9AED2"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fval(row: dict[str, str], key: str) -> float:
    return float(row[key])


def canonical_controller(label: str) -> str:
    if label in {"IR-CMPC", "VICM-Ac"}:
        return "IR-CMPC"
    if label in {"IRM-CMPC", "IR-CMPC-Hrel"}:
        return "IRM-CMPC"
    return label


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=INK,
        clip_on=False,
    )


def contiguous_boundary(
    row_map: dict[tuple[str, float, float], float],
    controller: str,
    directions: list[float],
    forces: list[float],
    threshold: float,
) -> list[tuple[float, float]]:
    boundary: list[tuple[float, float]] = []
    for direction in directions:
        max_force = 0.0
        for force in forces:
            if row_map.get((controller, direction, force), 0.0) < threshold:
                break
            max_force = force
        boundary.append((math.radians(direction), max_force))
    return boundary


def observed_max_boundary(
    row_map: dict[tuple[str, float, float], float],
    controller: str,
    directions: list[float],
    forces: list[float],
    threshold: float,
) -> list[tuple[float, float]]:
    boundary: list[tuple[float, float]] = []
    for direction in directions:
        recovered_forces = [
            force
            for force in forces
            if row_map.get((controller, direction, force), 0.0) >= threshold
        ]
        boundary.append(
            (math.radians(direction), max(recovered_forces, default=0.0))
        )
    return boundary


def polar_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    points = sorted(points)
    area = 0.0
    for index, (theta_i, radius_i) in enumerate(points):
        theta_j, radius_j = points[(index + 1) % len(points)]
        if index == len(points) - 1:
            theta_j += 2.0 * math.pi
        area += 0.5 * radius_i * radius_j * math.sin(theta_j - theta_i)
    return area


def plot(
    input_dir: Path,
    output: Path,
    lambda_scale: float,
    threshold: float,
    boundary_mode: str,
) -> None:
    rows = read_rows(input_dir / "summary.csv")
    directions = sorted({fval(row, "push_angle_deg") for row in rows})
    forces = sorted({fval(row, "push_force") for row in rows})
    row_map = {
        (
            canonical_controller(row["controller_label"]),
            fval(row, "push_angle_deg"),
            fval(row, "push_force"),
        ): fval(row, "success_rate")
        for row in rows
    }
    controllers = ["SRBM", "IR-CMPC", "IRM-CMPC"]
    missing = [
        key
        for controller in controllers
        for direction in directions
        for force in forces
        if (key := (controller, direction, force)) not in row_map
    ]
    if missing:
        raise RuntimeError(f"missing {len(missing)} controller-direction-force cells")

    boundary_builder = (
        observed_max_boundary
        if boundary_mode == "observed-max"
        else contiguous_boundary
    )
    boundaries = {
        controller: boundary_builder(
            row_map, controller, directions, forces, threshold
        )
        for controller in controllers
    }
    full_boundary = [
        (math.radians(direction), max(forces)) for direction in directions
    ]
    full_area = polar_area(full_boundary)
    area_fractions = {
        controller: polar_area(boundaries[controller]) / full_area
        for controller in controllers
    }

    set_style()
    fig = plt.figure(figsize=(3.45, 5.35))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.08, 0.92], hspace=0.36)
    polar = fig.add_subplot(grid[0, 0], projection="polar")
    for controller, color, fill_color, linestyle, title in [
        ("SRBM", SRBM, SRBM_FILL, "-", "SRBM"),
        ("IR-CMPC", IR, IR_FILL, "--", "IR-CMPC"),
        ("IRM-CMPC", IRM, IRM_FILL, "-.", "IRM-CMPC (Ours)"),
    ]:
        points = sorted(boundaries[controller])
        theta = np.asarray([point[0] for point in points] + [points[0][0]])
        radius = np.asarray([point[1] for point in points] + [points[0][1]])
        polar.fill(theta, radius, color=fill_color, alpha=0.34, linewidth=0, zorder=1)
        polar.plot(
            theta,
            radius,
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            marker="o",
            markersize=3,
            label=title,
            zorder=2,
        )
    polar.set_theta_zero_location("E")
    polar.set_theta_direction(1)
    polar.set_thetagrids(
        directions,
        labels=[rf"${direction:.0f}^\circ$" for direction in directions],
    )
    polar.set_rlabel_position(35)
    polar.set_rticks(forces[1:])
    polar.set_rlim(0, max(forces) * 1.05)
    polar.grid(color=GRID, linewidth=0.55)
    polar.spines["polar"].set_color(AXIS)
    polar.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.38),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    polar.text(
        0.50,
        1.22,
        rf"$\lambda={lambda_scale:.1f},\quad "
        rf"A_\mathrm{{IR}}/A_\mathrm{{SRBM}}="
        rf"{area_fractions['IR-CMPC'] / area_fractions['SRBM']:.3f},\quad "
        rf"A_\mathrm{{IRM}}/A_\mathrm{{SRBM}}="
        rf"{area_fractions['IRM-CMPC'] / area_fractions['SRBM']:.3f}$",
        transform=polar.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=INK,
        clip_on=False,
    )
    panel_label(polar, "(a)", x=-0.10, y=1.02)

    direction_ax = fig.add_subplot(grid[1, 0])
    direction_ax.set_aspect("equal", adjustable="box")
    direction_ax.set_xlim(-0.78, 1.02)
    direction_ax.set_ylim(-0.72, 0.82)
    direction_ax.axis("off")
    for patch in [
        Rectangle((-0.18, -0.14), 0.48, 0.28, linewidth=0.8, edgecolor=INK, facecolor="#EEF0F5"),
        Rectangle((-0.08, 0.28), 0.30, 0.08, linewidth=0.7, edgecolor=MUTED, facecolor="white"),
        Rectangle((-0.08, -0.36), 0.30, 0.08, linewidth=0.7, edgecolor=MUTED, facecolor="white"),
    ]:
        direction_ax.add_patch(patch)
    direction_ax.text(0.06, 0.0, "base", ha="center", va="center", color=INK)
    direction_ax.text(0.07, 0.43, "left foot", ha="center", va="bottom", color=MUTED)
    direction_ax.text(0.22, -0.39, "right foot", ha="left", va="center", color=MUTED)
    origin = (-0.58, -0.48)
    direction_ax.annotate("", xy=(0.45, origin[1]), xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK})
    direction_ax.annotate("", xy=(origin[0], 0.52), xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK})
    direction_ax.text(0.49, origin[1] - 0.05, r"$+X,\ v_x$", ha="left", va="top", color=INK)
    direction_ax.text(origin[0] - 0.03, 0.56, r"$+Y,\ v_y$", ha="right", va="bottom", color=INK)
    theta = math.radians(45.0)
    force_end = (
        origin[0] + 0.86 * math.cos(theta),
        origin[1] + 0.86 * math.sin(theta),
    )
    direction_ax.annotate("", xy=force_end, xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.25, "color": IR})
    direction_ax.text(force_end[0] + 0.02, force_end[1] + 0.02, r"$\mathbf{F}_{\mathrm{push}}$", ha="left", va="bottom", color=IR)
    direction_ax.add_patch(Arc(origin, 0.46, 0.46, theta1=0, theta2=45, linewidth=0.8, color=MUTED))
    direction_ax.text(origin[0] + 0.30, origin[1] + 0.08, r"$\theta_F$", ha="left", va="center", color=MUTED)
    direction_ax.text(
        0.98,
        0.98,
        r"$\theta_F=0^\circ:\ +X$" + "\n" + r"$\theta_F=90^\circ:\ +Y$",
        transform=direction_ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
    )
    panel_label(direction_ax, "(b)", x=-0.06, y=1.02)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"SRBM_AREA_FRACTION={area_fractions['SRBM']:.6f}")
    print(f"IR_CMPC_AREA_FRACTION={area_fractions['IR-CMPC']:.6f}")
    print(f"IRM_CMPC_AREA_FRACTION={area_fractions['IRM-CMPC']:.6f}")
    print(f"BOUNDARY_MODE={boundary_mode}")
    if area_fractions["SRBM"] > 1e-12:
        ir_change = area_fractions["IR-CMPC"] / area_fractions["SRBM"] - 1.0
        irm_change = area_fractions["IRM-CMPC"] / area_fractions["SRBM"] - 1.0
        print(f"IR_AREA_RELATIVE_CHANGE={ir_change:.6f}")
        print(f"IRM_AREA_RELATIVE_CHANGE={irm_change:.6f}")
    print(f"PNG={output.with_suffix('.png')}")
    print(f"PDF={output.with_suffix('.pdf')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lambda-scale", type=float, default=1.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--boundary-mode",
        choices=["observed-max", "contiguous"],
        default="observed-max",
    )
    args = parser.parse_args()
    plot(
        args.input_dir,
        args.output,
        args.lambda_scale,
        args.threshold,
        args.boundary_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
