#!/usr/bin/env python3
"""Capture side-view flat-ground walking snapshots for Fig. 4.

The executable saves raw PPM frames through the ODC_SNAPSHOT_* environment
variables. This script runs a nominal straight-walking trial and assembles the
10 frames into a publication figure.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vicm_experiment_lib import BUILD_DIR, BIN, MPC_L_DIAG_MAIN, REPO_ROOT, make_env


OUT_FIG = REPO_ROOT / "figures" / "manuscript_current" / "fig4_walking_snapshots.png"
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
LEG_STATE_LABELS = {
    0: "Left support",
    1: "Right support",
    2: "Double support",
}


def token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def ensure_build(skip_build: bool) -> None:
    if skip_build and BIN.exists():
        return
    if not (REPO_ROOT / "build" / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)], check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j4"], check=True)


def center_crop_width(image: Image.Image, crop_width: int) -> Image.Image:
    width, height = image.size
    crop_width = min(crop_width, width)
    left = max(0, (width - crop_width) // 2)
    return image.crop((left, 0, left + crop_width, height))


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def support_transitions(trace_path: Path, t0: float, t1: float) -> list[float]:
    if not trace_path.exists():
        return []
    transitions: list[float] = []
    previous_state: int | None = None
    with trace_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time"])
            if t < t0:
                previous_state = int(float(row["leg_state"]))
                continue
            if t > t1:
                break
            state = int(float(row["leg_state"]))
            if previous_state is not None and state != previous_state:
                transitions.append(t)
            previous_state = state
    return transitions


def support_segments(trace_path: Path, t0: float, t1: float) -> list[tuple[float, float, int]]:
    if not trace_path.exists():
        return []
    samples: list[tuple[float, int]] = []
    with trace_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time"])
            if t < t0:
                samples = [(t0, int(float(row["leg_state"])))]
                continue
            if t > t1:
                break
            samples.append((t, int(float(row["leg_state"]))))

    if not samples:
        return []

    segments: list[tuple[float, float, int]] = []
    state = samples[0][1]
    start = t0
    for t, next_state in samples[1:]:
        if next_state != state:
            segments.append((start, t, state))
            start = t
            state = next_state
    segments.append((start, t1, state))
    return [(a, b, s) for a, b, s in segments if b - a > 1e-4]


def parse_times(text: str) -> list[float]:
    times = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot times must be strictly increasing")
    return times


def sampled_support_states(
    trace_path: Path,
    sample_times: list[float],
    double_support_threshold: float,
) -> list[int]:
    if not trace_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with trace_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    trace_times = [float(row["time"]) for row in rows]
    states: list[int] = []
    cursor = 0
    for sample_time in sample_times:
        while cursor + 1 < len(trace_times) and abs(trace_times[cursor + 1] - sample_time) <= abs(trace_times[cursor] - sample_time):
            cursor += 1
        row = rows[cursor]
        leg_state = int(float(row["leg_state"]))
        fz_l = float(row.get("fLz_touch", "nan"))
        fz_r = float(row.get("fRz_touch", "nan"))
        if fz_l >= double_support_threshold and fz_r >= double_support_threshold:
            states.append(2)
        else:
            states.append(leg_state)
    return states


def support_runs(states: list[int]) -> list[tuple[int, int, int]]:
    if not states:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    state = states[0]
    for idx, next_state in enumerate(states[1:], start=1):
        if next_state != state:
            runs.append((start, idx, state))
            start = idx
            state = next_state
    runs.append((start, len(states), state))
    return runs


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: int, height: int) -> None:
    dash_len = 13
    gap_len = 12
    y = 0
    while y < height:
        draw.line(
            (x, y, x, min(height, y + dash_len)),
            fill=(255, 255, 255, 185),
            width=2,
        )
        y += dash_len + gap_len


def draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] = (255, 255, 255, 235),
) -> None:
    pad_x, pad_y = 7, 5
    text_w, text_h = text_size(draw, text, font)
    x0 = int(center_x - text_w / 2 - pad_x)
    y0 = y
    x1 = int(center_x + text_w / 2 + pad_x)
    y1 = y + text_h + 2 * pad_y
    if x0 < 4:
        x1 += 4 - x0
        x0 = 4
    if x1 > draw.im.size[0] - 4:
        x0 -= x1 - (draw.im.size[0] - 4)
        x1 = draw.im.size[0] - 4
    text_x = int(round((x0 + x1 - text_w) / 2))
    draw.rectangle((x0, y0, x1, y1), fill=(12, 28, 42, 135), outline=(255, 255, 255, 100))
    draw.text((text_x, y + pad_y), text, fill=fill, font=font)


def assemble_montage(
    frame_paths: list[Path],
    out_path: Path,
    sample_times: list[float],
    trace_path: Path,
    double_support_threshold: float,
) -> None:
    if len(frame_paths) != 10:
        raise RuntimeError(f"expected 10 snapshot frames, found {len(frame_paths)}")
    if len(sample_times) != len(frame_paths):
        raise RuntimeError(f"expected {len(frame_paths)} sample times, found {len(sample_times)}")

    raw_crop_w = 300
    tile_h = 360
    raw_h = Image.open(frame_paths[0]).height
    tile_w = int(round(tile_h * raw_crop_w / raw_h))
    canvas_w = tile_w * len(frame_paths)
    canvas_h = tile_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for idx, path in enumerate(frame_paths):
        img = Image.open(path).convert("RGB")
        img = center_crop_width(img, raw_crop_w).resize((tile_w, tile_h), RESAMPLE_LANCZOS)
        canvas.paste(img, (idx * tile_w, 0))

    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    label_font = load_font(22)

    states = sampled_support_states(trace_path, sample_times, double_support_threshold)
    for start_idx, end_idx, state in support_runs(states):
        label = LEG_STATE_LABELS.get(state, f"State {state}")
        x0 = start_idx * tile_w
        x1 = end_idx * tile_w
        draw_label(draw, label, int(round((x0 + x1) / 2)), 18, label_font)

    for boundary_idx in range(1, len(states)):
        if states[boundary_idx] == states[boundary_idx - 1]:
            continue
        draw_dashed_vertical(draw, boundary_idx * tile_w, canvas_h)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vx", type=float, default=1.0, help="straight walking command speed [m/s]")
    parser.add_argument("--tswing", type=float, default=0.45, help="swing duration [s]")
    parser.add_argument("--start", type=float, default=10.50, help="first snapshot time [s]")
    parser.add_argument("--interval", type=float, default=0.095, help="snapshot interval [s]")
    parser.add_argument(
        "--times",
        default="3.06,3.22,3.38,3.54,3.66,3.78,3.94,4.08,4.22,4.35",
        help="comma-separated nonuniform snapshot times [s]",
    )
    parser.add_argument(
        "--double-support-threshold",
        type=float,
        default=150.0,
        help="foot vertical-force threshold used to label double support [N]",
    )
    parser.add_argument("--count", type=int, default=10, help="number of snapshots")
    parser.add_argument("--skip-build", action="store_true", help="do not rebuild if the binary already exists")
    parser.add_argument("--out", type=Path, default=OUT_FIG, help="assembled output figure path")
    args = parser.parse_args()

    if args.count != 10:
        raise ValueError("Fig. 4 montage is designed for exactly 10 snapshots")
    sample_times = parse_times(args.times) if args.times else [
        args.start + i * args.interval for i in range(args.count)
    ]
    if len(sample_times) != args.count:
        raise ValueError("--times must contain exactly --count values")

    ensure_build(args.skip_build)

    run_dir = REPO_ROOT / "record" / (
        f"fig4_walking_snapshots_vx{token(args.vx)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "walk_snapshot_run.log"

    env, _ = make_env(
        exp_id=2,
        case="fig4_walking_snapshots",
        controller="vicm_ac",
        sim_end=sample_times[-1] + 0.5,
        vx=args.vx,
        vy=0.0,
        tswing=args.tswing,
        posrot_att_scale=0.35,
        posrot_pos_scale=1.0,
        tau_bias_scale=1.0,
        tau_non_norm_limit=0.0,
        ig_dot_filter_tau=0.01,
        mpc_l_diag=MPC_L_DIAG_MAIN,
        torque_limit_scale=1.2,
        walk_leg_pd_scale=1.2,
        lambda_scale=1.0,
        sine_turn=False,
        sine_wz_amp=0.0,
        gait_switch_threshold=100.0,
    )
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_LOG_PREDICTION_ERROR": "0",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_PRINT_GAIT_SWITCH": "0",
            "ODC_SNAPSHOT_ENABLE": "1",
            "ODC_SNAPSHOT_DIR": str(frame_dir),
            "ODC_SNAPSHOT_PREFIX": "fig4_walking_snapshot",
            "ODC_SNAPSHOT_START_TIME": f"{args.start:.12g}",
            "ODC_SNAPSHOT_INTERVAL": f"{args.interval:.12g}",
            "ODC_SNAPSHOT_COUNT": str(args.count),
            "ODC_SNAPSHOT_TIMES": ",".join(f"{time:.12g}" for time in sample_times),
            "ODC_SNAPSHOT_EXIT_AFTER_CAPTURE": "1",
        }
    )

    with log_path.open("w") as log:
        subprocess.run(
            [str(BIN)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    trace_path = REPO_ROOT / "record" / "exp2_trace.csv"
    frame_paths = sorted(frame_dir.glob("fig4_walking_snapshot_*.ppm"))
    assemble_montage(frame_paths, args.out, sample_times, trace_path, args.double_support_threshold)
    shutil.copyfile(args.out, run_dir / args.out.name)
    if trace_path.exists():
        shutil.copyfile(trace_path, run_dir / "fig4_walking_snapshot_trace.csv")

    print(f"Saved montage: {args.out}")
    print(f"Raw frames: {frame_dir}")
    print(f"Run log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
