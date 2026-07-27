#!/usr/bin/env python3
"""Re-render final experiment figures in a compact paper style.

This script is intentionally data-only: it reads accepted experiment logs and
overwrites the final figure files used by the manuscript.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import Arc, PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "manuscript_current"

EXP1_DIR = ROOT / "record" / "lambda_filter_turn_exp1_20260722_232307"
EXP1_RETRY_DIRS: dict[float, Path] = {}
EXP3_DIR = ROOT / "record" / "exp3_model_ablation_lam1p8_20260722_211126"
EXP3_NF_RETRY_DIR = EXP3_DIR
EXP4_DIR = ROOT / "record" / "exp4_push_recovery_lam1p7_20260611_224747"

INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"
SRBM = "#2E4780"
VICM = "#CC6F47"
VICM_LABEL = "IR-CMPC (Ours)"
REF = "#464C55"
OLIVE = "#71B436"
PINK = "#BD569B"
GOLD = "#B8A037"
NEUTRAL = "#7A828F"

MODEL_COLORS = {
    "SRBM": SRBM,
    "VICM": VICM,
    VICM_LABEL: VICM,
    "VICM-Ac": VICM,
    "VI-CMPC": OLIVE,
    "IR-CMPC-NF": PINK,
    "VICM-IG": OLIVE,
    "VICM-Ig": OLIVE,
    "VICM-NF": PINK,
    "VICM-Ac no filter": PINK,
    "VICM affine tau": GOLD,
}


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.65,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def trace_arrays(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {key: [] for key in keys}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in keys:
                data[key].append(fval(row, key))
    return {key: np.asarray(vals, dtype=float) for key, vals in data.items()}


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y
    window = min(window, len(y))
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_by_time(t: np.ndarray, y: np.ndarray, seconds: float = 0.2) -> np.ndarray:
    if len(t) < 3:
        return y
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return y
    return smooth(y, max(1, int(round(seconds / dt))))


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="normal",
        color=INK,
        clip_on=False,
    )


def bottom_label(ax: plt.Axes, label: str, y: float = -0.22) -> None:
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=INK,
        clip_on=False,
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(path.relative_to(ROOT))


def add_top_brace(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    height: float,
    label: str,
) -> None:
    """Draw a compact horizontal curly brace in data coordinates."""
    dx = x1 - x0
    mid = 0.5 * (x0 + x1)
    verts = [
        (x0, y),
        (x0 + 0.03 * dx, y),
        (x0 + 0.05 * dx, y + height),
        (x0 + 0.12 * dx, y + height),
        (x0 + 0.28 * dx, y + height),
        (mid - 0.08 * dx, y + height),
        (mid - 0.035 * dx, y + 0.08 * height),
        (mid - 0.015 * dx, y - 0.18 * height),
        (mid - 0.005 * dx, y - 0.18 * height),
        (mid, y - 0.18 * height),
        (mid + 0.005 * dx, y - 0.18 * height),
        (mid + 0.015 * dx, y - 0.18 * height),
        (mid + 0.035 * dx, y + 0.08 * height),
        (mid + 0.08 * dx, y + height),
        (x1 - 0.28 * dx, y + height),
        (x1 - 0.12 * dx, y + height),
        (x1 - 0.05 * dx, y + height),
        (x1 - 0.03 * dx, y),
        (x1, y),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(verts) - 1)
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor="none",
            edgecolor=MUTED,
            lw=0.75,
            capstyle="round",
            joinstyle="round",
            clip_on=False,
        )
    )
    ax.text(
        mid,
        y + height + 0.32,
        label,
        ha="center",
        va="bottom",
        fontsize=6.7,
        color=INK,
        clip_on=False,
    )


def plot_exp1_survival() -> None:
    rows = read_rows(EXP1_DIR / "summary.csv")
    for lam, retry_dir in EXP1_RETRY_DIRS.items():
        rows = [row for row in rows if not math.isclose(fval(row, "lambda_scale"), lam)]
        rows.extend(read_rows(retry_dir / "summary.csv"))
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        lam = fval(row, "lambda_scale")
        if lam < 1.0:
            continue
        ctrl = "SRBM" if row["controller"] == "srbm" else "VICM"
        grouped[ctrl].append(
            (lam, fval(row, "mean_final_time"), fval(row, "std_final_time", 0.0))
        )

    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    regions = [
        (0.98, 1.75, r"(a) Low/mid", "#EEF1F6"),
        (1.75, 2.05, r"(b) Advantage", "#E8F0FA"),
        (2.05, 2.32, r"(c) High", "#EEF1F6"),
    ]
    for x0, x1, _, color in regions:
        ax.axvspan(x0, x1, color=color, alpha=0.55, zorder=0)
    for x in [1.75, 2.05]:
        ax.axvline(x, color=AXIS, linewidth=0.6, linestyle="--", zorder=1)

    for ctrl, marker, linestyle in [("SRBM", "o", "-"), ("VICM", "s", "--")]:
        pts = sorted(grouped[ctrl])
        x = np.asarray([p[0] for p in pts])
        y = np.asarray([p[1] for p in pts])
        yerr = np.asarray([p[2] for p in pts])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=MODEL_COLORS[ctrl],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=3.2,
            elinewidth=0.75,
            capsize=2.0,
            capthick=0.75,
            zorder=4,
            label=VICM_LABEL if ctrl == "VICM" else ctrl,
        )

    ax.axhline(30.0, color=REF, linewidth=0.8, linestyle=":")
    ax.text(
        2.30,
        28.45,
        "30 s limit",
        ha="right",
        va="top",
        fontsize=6.6,
        color=REF,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.78},
    )
    for x0, x1, label, _ in regions:
        add_top_brace(ax, x0 + 0.02, x1 - 0.02, 31.2, 0.42, label)

    mass_note = (
        r"$M_b^0=47.48$ kg" "\n"
        r"$M_\ell^0=29.87$ kg" "\n"
        r"$M_\ell(\lambda)=\lambda M_\ell^0$"
    )
    ax.text(
        1.03,
        2.7,
        mass_note,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=INK,
        bbox={
            "facecolor": "white",
            "edgecolor": AXIS,
            "linewidth": 0.45,
            "pad": 2.0,
            "alpha": 0.86,
        },
    )
    ax.text(
        1.90,
        4.7,
        "IR-CMPC\nadvantage window",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MODEL_COLORS["VICM"],
        fontweight="semibold",
        bbox={
            "facecolor": "white",
            "edgecolor": MODEL_COLORS["VICM"],
            "linewidth": 0.55,
            "pad": 2.0,
            "alpha": 0.82,
        },
        zorder=5,
    )
    ax.set_xlabel(r"Leg inertia scale $\lambda$")
    ax.set_ylabel("Survival time [s]")
    ax.set_xlim(0.98, 2.32)
    ax.set_ylim(0.0, 33.0)
    ax.set_xticks(np.arange(1.0, 2.31, 0.2))
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.32), frameon=False, ncol=1, handlelength=2.2)
    save(fig, OUT_DIR / "exp1_lambda_turning_survival.png")


def plot_exp1_tracking() -> None:
    cases = [
        (
            "lam1",
            r"$\lambda=1.0$",
            EXP1_DIR / "lam1_srbm_turn_posrot0p35_filtertau_r1_trace.csv",
            EXP1_DIR / "lam1_vicm_turn_posrot0p35_filtertau_r1_trace.csv",
            "(a)",
        ),
        (
            "lam1p8",
            r"$\lambda=1.8$",
            EXP1_DIR / "lam1p8_srbm_turn_posrot0p35_filtertau_r1_trace.csv",
            EXP1_DIR / "lam1p8_vicm_turn_posrot0p35_filtertau_r1_trace.csv",
            "(b)",
        ),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.25), sharex=False, sharey=True)
    for idx, (ax, (_, lambda_label, srbm_path, vicm_path, label)) in enumerate(zip(axes, cases)):
        traces = [
            ("SRBM", trace_arrays(srbm_path, ["time", "vx", "vx_ref"]), SRBM, "-"),
            (VICM_LABEL, trace_arrays(vicm_path, ["time", "vx", "vx_ref"]), VICM, "--"),
        ]
        fail_points: dict[str, tuple[float, float]] = {}
        for ctrl, data, color, linestyle in traces:
            t = data["time"]
            y_smooth = smooth_by_time(t, data["vx"], 0.15)
            ax.plot(
                t,
                y_smooth,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=ctrl,
            )
            if idx == 1 and len(t) > 0 and float(t[-1]) < 29.9:
                fail_points[ctrl] = (float(t[-1]), float(y_smooth[-1]))
        ref = traces[-1][1]
        ax.plot(ref["time"], ref["vx_ref"], color=REF, linewidth=0.8, linestyle=":", label="Reference")
        ax.set_xlim(0, 30)
        ax.set_ylim(-0.2, 1.75)
        ax.set_ylabel(r"$v_x$ [m/s]")
        ax.set_xlabel("Time [s]")
        panel_label(ax, label, x=-0.18, y=1.08)
        ax.text(
            0.04,
            0.80,
            lambda_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK,
            clip_on=True,
        )
        if "SRBM" in fail_points:
            fail_t, fail_v = fail_points["SRBM"]
            ax.annotate(
                "SRBM fail",
                xy=(fail_t, fail_v),
                xytext=(fail_t + 1.75, 0.34),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.75,
                    "color": SRBM,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                ha="left",
                va="center",
                fontsize=6.8,
                color=SRBM,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.78},
            )
        if VICM_LABEL in fail_points:
            fail_t, fail_v = fail_points[VICM_LABEL]
            ax.annotate(
                "IR-CMPC fail",
                xy=(fail_t, fail_v),
                xytext=(fail_t + 1.2, 0.73),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.75,
                    "color": VICM,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                ha="left",
                va="center",
                fontsize=6.8,
                color=VICM,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.78},
            )
        clean_axes(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=3,
        columnspacing=0.9,
        handlelength=1.8,
    )
    fig.subplots_adjust(hspace=0.55, bottom=0.12, top=0.88)
    save(fig, OUT_DIR / "exp1_representative_tracking_body_forward.png")


def plot_exp3_ablation() -> None:
    rows = read_rows(EXP3_DIR / "summary.csv")
    by_label = {row["controller_label"]: row for row in rows}
    label_map = [
        ("SRBM", "SRBM"),
        ("VICM-Ig", "VI-CMPC"),
        ("VICM-Ac", VICM_LABEL),
        ("VICM-Ac no filter", "IR-CMPC-NF"),
    ]
    pairs = [(raw, shown) for raw, shown in label_map if raw in by_label]
    labels = [shown for _, shown in pairs]

    specs = [
        ("mean_final_time", "Survival time [s]", "(a)", (0, 32)),
        ("mean_rms_wz_err", r"RMS error of $\omega_z$ [rad/s]", "(b)", None),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.75))
    y = np.arange(len(labels))
    for idx, (ax, (field, xlabel, label, xlim)) in enumerate(zip(axes[:2], specs)):
        vals = [fval(by_label[raw], field) for raw, _ in pairs]
        err_field = field.replace("mean_", "std_", 1)
        errs = [fval(by_label[raw], err_field, 0.0) for raw, _ in pairs]
        colors = [MODEL_COLORS.get(item, NEUTRAL) for item in labels]
        ax.barh(y, vals, xerr=errs, color=colors, edgecolor="white", linewidth=0.5, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, labelpad=8)
        panel_label(ax, label)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(0.0, max(value + error for value, error in zip(vals, errs)) * 1.28)
        if idx == 1 and "SRBM" in labels and VICM_LABEL in labels:
            srbm_value = vals[labels.index("SRBM")]
            vicm_index = labels.index(VICM_LABEL)
            vicm_value = vals[vicm_index]
            reduction = 100.0 * (srbm_value - vicm_value) / srbm_value
            comparison = (
                f"{reduction:.1f}% lower"
                if reduction >= 0.0
                else f"{abs(reduction):.1f}% higher"
            )
            ax.annotate(
                comparison,
                xy=(vicm_value + errs[vicm_index], y[vicm_index]),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.4,
                color=MODEL_COLORS[VICM_LABEL],
                fontweight="semibold",
            )
        if idx == 0:
            ax.set_ylim(len(labels) - 0.5, -0.95)
            ax.text(
                31.6,
                -0.62,
                r"$\lambda=1.8$",
                ha="right",
                va="center",
                fontsize=7,
                color=INK,
            )
        clean_axes(ax, grid_axis="x")

    trial_rows = read_rows(EXP3_DIR / "trials.csv")
    prediction_specs = [
        ("rms_srbm_pred_err", "SRBM"),
        ("rms_vi_pred_err", "VI-CMPC"),
        ("rms_ir_pred_err", VICM_LABEL),
        ("rms_ir_nf_pred_err", "IR-CMPC-NF"),
    ]
    pred_values = []
    pred_errors = []
    pred_labels = []
    for field, label in prediction_specs:
        samples = np.asarray([fval(row, field) for row in trial_rows], dtype=float)
        samples = samples[np.isfinite(samples)]
        pred_values.append(float(np.mean(samples)))
        pred_errors.append(float(np.std(samples)))
        pred_labels.append(label)
    ax = axes[2]
    pred_y = np.arange(len(pred_labels))
    ax.barh(
        pred_y,
        pred_values,
        xerr=pred_errors,
        color=[MODEL_COLORS.get(item, NEUTRAL) for item in pred_labels],
        edgecolor="white",
        linewidth=0.5,
        height=0.68,
    )
    ax.set_yticks(pred_y)
    ax.set_yticklabels(pred_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(v + e for v, e in zip(pred_values, pred_errors)) * 1.18)
    ax.set_xlabel(r"One-step RMS error of $\omega$ [rad/s]", labelpad=8)
    panel_label(ax, "(c)")
    clean_axes(ax, grid_axis="x")
    fig.subplots_adjust(hspace=0.70)
    save(fig, OUT_DIR / "exp3_model_ablation_summary.png")


def plot_exp4_recovery_heatmap() -> None:
    rows = read_rows(EXP4_DIR / "summary.csv")
    controllers = ["SRBM", "VICM-Ac"]
    directions = sorted({fval(row, "push_angle_deg") for row in rows})
    plot_directions = list(reversed(directions))
    forces = sorted({fval(row, "push_force") for row in rows})
    row_map = {
        (row["controller_label"], fval(row, "push_angle_deg"), fval(row, "push_force")): fval(row, "success_rate")
        for row in rows
    }
    cmap = LinearSegmentedColormap.from_list("recovery", ["#F4F5F7", "#5477C4"])

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.85), sharey=True)
    im = None
    for ax, ctrl, label in zip(axes, controllers, ["(a)", "(b)"]):
        matrix = np.asarray(
            [
                [row_map.get((ctrl, direction, force), math.nan) for force in forces]
                for direction in plot_directions
            ],
            dtype=float,
        )
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
        panel_label(ax, label, x=-0.10, y=1.08)
        ax.set_xticks(np.arange(len(forces)))
        ax.set_xticklabels([f"{force:.0f}" for force in forces])
        ax.set_yticks(np.arange(len(plot_directions)))
        ax.set_yticklabels([f"{direction:.0f}" for direction in plot_directions])
        ax.set_xlabel("Push force [N]")
        ax.tick_params(length=0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > 0.55 else INK,
                    )
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(0.65)
    axes[0].set_ylabel("Push direction [deg]")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Recovery rate")
    cbar.outline.set_linewidth(0.5)
    save(fig, OUT_DIR / "exp4_push_recovery_heatmap.png")


def polar_area(points: list[tuple[float, float]]) -> float:
    pts = sorted(points)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i, (theta_i, radius_i) in enumerate(pts):
        theta_j, radius_j = pts[(i + 1) % len(pts)]
        if i == len(pts) - 1:
            theta_j += 2.0 * math.pi
        area += 0.5 * radius_i * radius_j * math.sin(theta_j - theta_i)
    return area


def plot_exp4_binary_recovery_and_boundary() -> None:
    rows = read_rows(EXP4_DIR / "summary.csv")
    boundary_rows = read_rows(EXP4_DIR / "recovery_boundary.csv")
    controllers = ["SRBM", "VICM-Ac"]
    directions = sorted({fval(row, "push_angle_deg") for row in rows})
    forces = sorted({fval(row, "push_force") for row in rows})
    row_map = {
        (row["controller_label"], fval(row, "push_angle_deg"), fval(row, "push_force")): fval(row, "success_rate")
        for row in rows
    }
    by_ctrl: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in boundary_rows:
        by_ctrl[row["controller_label"]].append(
            (math.radians(fval(row, "push_angle_deg")), fval(row, "max_recoverable_force"))
        )
    area_srbm = polar_area(by_ctrl.get("SRBM", []))
    area_vicm = polar_area(by_ctrl.get("VICM-Ac", []))
    area_full = polar_area(
        [(math.radians(direction), max(forces)) for direction in directions]
    )
    area_fraction_srbm = area_srbm / area_full if area_full > 1e-9 else math.nan
    area_fraction_vicm = area_vicm / area_full if area_full > 1e-9 else math.nan

    fig = plt.figure(figsize=(6.8, 5.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], hspace=0.72, wspace=0.30)
    binary_cmap = ListedColormap(["white", SRBM])
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, ctrl, label in zip(top_axes, controllers, ["(a)", "(b)"]):
        matrix = np.asarray(
            [
                [row_map.get((ctrl, direction, force), math.nan) for force in forces]
                for direction in directions
            ],
            dtype=float,
        )
        binary = np.where(matrix >= 0.5, 1.0, 0.0)
        ax.imshow(binary, vmin=0.0, vmax=1.0, cmap=binary_cmap, aspect="auto", origin="lower")
        panel_label(ax, label, x=-0.10, y=1.08)
        ax.set_xticks(np.arange(len(forces)))
        ax.set_xticklabels([f"{force:.0f}" for force in forces])
        ax.set_yticks(np.arange(len(directions)))
        ax.set_yticklabels([f"{direction:.0f}" for direction in directions])
        ax.set_xticks(np.arange(-0.5, len(forces), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(directions), 1), minor=True)
        ax.grid(which="minor", color="#CBD0DA", linewidth=0.55)
        ax.tick_params(which="both", length=0)
        ax.set_xlabel("Push force [N]")
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(0.65)
    top_axes[0].set_ylabel(r"Push direction $\theta_F$ [deg]")
    top_axes[1].set_yticklabels([])
    top_axes[0].text(
        0.02,
        1.22,
        rf"$\lambda=1.7$",
        transform=top_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=INK,
    )
    for ax, title, fraction, color in [
        (top_axes[0], "SRBM", area_fraction_srbm, SRBM),
        (top_axes[1], VICM_LABEL, area_fraction_vicm, VICM),
    ]:
        area_symbol = r"A_\mathrm{SRBM}" if title == "SRBM" else r"A_\mathrm{IR}"
        ax.text(
            0.50,
            1.10,
            title
            + rf": ${area_symbol}/A_\mathrm{{full}}={fraction:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=color,
            fontweight="semibold",
        )
    legend_y = 1.22
    for x0, face, edge, label in [
        (0.48, SRBM, SRBM, "Recovered"),
        (0.78, "white", AXIS, "Failed"),
    ]:
        rect = Rectangle(
            (x0, legend_y - 0.038),
            0.045,
            0.045,
            transform=top_axes[1].transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.55,
            clip_on=False,
        )
        top_axes[1].add_patch(rect)
        top_axes[1].text(
            x0 + 0.055,
            legend_y - 0.016,
            label,
            transform=top_axes[1].transAxes,
            ha="left",
            va="center",
            fontsize=6.6,
            color=MUTED,
        )

    ax = fig.add_subplot(gs[1, 0], projection="polar")
    for ctrl, color, linestyle in [("SRBM", SRBM, "-"), ("VICM-Ac", VICM, "--")]:
        pts = sorted(by_ctrl.get(ctrl, []))
        if not pts:
            continue
        theta = np.asarray([p[0] for p in pts] + [pts[0][0]])
        radius = np.asarray([p[1] for p in pts] + [pts[0][1]])
        label = VICM_LABEL if ctrl == "VICM-Ac" else ctrl
        ax.plot(theta, radius, color=color, linestyle=linestyle, linewidth=1.15, marker="o", markersize=3, label=label)
        ax.fill(theta, radius, color=color, alpha=0.09)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=[
            r"$0^\circ$",
            r"$45^\circ$",
            r"$90^\circ$",
            r"$135^\circ$",
            r"$180^\circ$",
            r"$225^\circ$",
            r"$270^\circ$",
            r"$315^\circ$",
        ],
    )
    ax.set_rlabel_position(35)
    ax.set_rticks([100, 200, 300, 400])
    ax.set_rlim(0, 420)
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines["polar"].set_color(AXIS)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.36),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    panel_label(ax, "(c)", x=-0.10, y=1.08)

    axd = fig.add_subplot(gs[1, 1])
    axd.set_aspect("equal", adjustable="box")
    axd.set_xlim(-0.78, 1.02)
    axd.set_ylim(-0.72, 0.82)
    axd.axis("off")

    # Top-view robot silhouette: +X is forward, +Y is robot-left/world-left.
    body = Rectangle((-0.18, -0.14), 0.48, 0.28, linewidth=0.8, edgecolor=INK, facecolor="#EEF0F5")
    left_foot = Rectangle((-0.08, 0.28), 0.30, 0.08, linewidth=0.7, edgecolor=MUTED, facecolor="white")
    right_foot = Rectangle((-0.08, -0.36), 0.30, 0.08, linewidth=0.7, edgecolor=MUTED, facecolor="white")
    for patch in [body, left_foot, right_foot]:
        axd.add_patch(patch)
    axd.text(0.06, 0.0, "base", ha="center", va="center", fontsize=7, color=INK)
    axd.text(0.07, 0.43, "left foot", ha="center", va="bottom", fontsize=6.8, color=MUTED)
    axd.text(0.22, -0.39, "right foot", ha="left", va="center", fontsize=6.8, color=MUTED)

    origin = (-0.58, -0.48)
    axd.annotate("", xy=(0.45, origin[1]), xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK})
    axd.annotate("", xy=(origin[0], 0.52), xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK})
    axd.text(0.49, origin[1] - 0.05, r"$+X,\ v_x$", ha="left", va="top", fontsize=7.5, color=INK)
    axd.text(origin[0] - 0.03, 0.56, r"$+Y,\ v_y$", ha="right", va="bottom", fontsize=7.5, color=INK)

    theta = math.radians(45.0)
    force_end = (origin[0] + 0.86 * math.cos(theta), origin[1] + 0.86 * math.sin(theta))
    axd.annotate("", xy=force_end, xytext=origin, arrowprops={"arrowstyle": "->", "lw": 1.25, "color": VICM})
    axd.text(force_end[0] + 0.02, force_end[1] + 0.02, r"$\mathbf{F}_{\mathrm{push}}$", ha="left", va="bottom", fontsize=7.5, color=VICM)
    axd.add_patch(Arc(origin, 0.46, 0.46, theta1=0, theta2=45, linewidth=0.8, color=MUTED))
    axd.text(origin[0] + 0.30, origin[1] + 0.08, r"$\theta_F$", ha="left", va="center", fontsize=7.5, color=MUTED)
    axd.text(
        0.98,
        0.98,
        r"$\theta_F=0^\circ:\ +X$" + "\n" + r"$\theta_F=90^\circ:\ +Y$",
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    panel_label(axd, "(d)", x=-0.10, y=1.08)
    save(fig, OUT_DIR / "exp4_recovery_binary_and_boundary.png")


def plot_exp4_polar_boundary() -> None:
    rows = read_rows(EXP4_DIR / "recovery_boundary.csv")
    by_ctrl: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_ctrl[row["controller_label"]].append(
            (math.radians(fval(row, "push_angle_deg")), fval(row, "max_recoverable_force"))
        )

    fig = plt.figure(figsize=(3.2, 2.9))
    ax = fig.add_subplot(111, projection="polar")
    for ctrl, color, linestyle in [("SRBM", SRBM, "-"), ("VICM-Ac", VICM, "--")]:
        pts = sorted(by_ctrl.get(ctrl, []))
        if not pts:
            continue
        theta = np.asarray([p[0] for p in pts] + [pts[0][0]])
        radius = np.asarray([p[1] for p in pts] + [pts[0][1]])
        ax.plot(theta, radius, color=color, linestyle=linestyle, linewidth=1.15, marker="o", markersize=3, label=VICM_LABEL if ctrl == "VICM-Ac" else ctrl)
        ax.fill(theta, radius, color=color, alpha=0.09)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=[
            r"$0^\circ$",
            r"$45^\circ$",
            r"$90^\circ$",
            r"$135^\circ$",
            r"$180^\circ$",
            r"$225^\circ$",
            r"$270^\circ$",
            r"$315^\circ$",
        ],
    )
    ax.set_rlabel_position(35)
    ax.set_rticks([100, 200, 300, 400])
    ax.set_rlim(0, 420)
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines["polar"].set_color(AXIS)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), frameon=False, handlelength=2.0)
    save(fig, OUT_DIR / "exp4_push_recovery_polar_boundary.png")


def plot_exp4_lateral_response() -> None:
    srbm = trace_arrays(
        EXP4_DIR / "exp4_lam1p7_dir90_F400_srbm_r1_trace.csv",
        ["time", "base_y", "vy", "torso_angle_error", "push_active"],
    )
    vicm = trace_arrays(
        EXP4_DIR / "exp4_lam1p7_dir90_F400_vicm_ac_r1_trace.csv",
        ["time", "base_y", "vy", "torso_angle_error", "push_active"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.1), sharex=True)
    configs = [
        (axes[0], "torso_angle_error", "Torso error [rad]", "(a)"),
        (axes[1], "base_y", "Lateral displacement [m]", "(b)"),
    ]
    for ax, key, ylabel, label in configs:
        for ctrl, data, color, linestyle in [
            ("SRBM", srbm, SRBM, "-"),
            (VICM_LABEL, vicm, VICM, "--"),
        ]:
            t = data["time"]
            y = data[key]
            if key == "base_y":
                y = y - y[0]
            ax.plot(t, smooth_by_time(t, y, 0.08), color=color, linestyle=linestyle, linewidth=1.05, label=ctrl)
        push_t = srbm["time"][srbm["push_active"] > 0.5]
        if len(push_t) > 0:
            ax.axvspan(push_t[0], push_t[-1], color="#E2E5EA", alpha=0.8, linewidth=0)
        ax.set_xlim(7.2, 14.2)
        ax.set_ylabel(ylabel)
        clean_axes(ax)
        panel_label(ax, label)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, handlelength=1.8)
    fig.subplots_adjust(hspace=0.18)
    save(fig, OUT_DIR / "exp4_lateral_push_response.png")


def main() -> None:
    set_ieee_style()
    plot_exp1_survival()
    plot_exp1_tracking()
    plot_exp3_ablation()
    plot_exp4_binary_recovery_and_boundary()


if __name__ == "__main__":
    main()
