from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIB_PATH = PROJECT_ROOT / "config" / "data_calib" / "calib.txt"
LEFT_DIR_NAME = "Left_picture"
RIGHT_DIR_NAME = "Right_picture"
TWO_PI = 2.0 * np.pi
POINT_CLOUD_Z_MIN_MM = 300.0
POINT_CLOUD_Z_MAX_MM = 500.0
POINT_CLOUD_MIN_COMPONENT_PIXELS = 64
MIN_PHASE_MODULATION = 8.0
MIN_PHASE_INTENSITY_SWING = 16.0
PHASE_MODULATION_RELATIVE_THRESHOLD = 0.15
PHASE_SWING_RELATIVE_THRESHOLD = 0.12
PHASE_JUMP_THRESHOLD = 1.5 * TWO_PI
PIXEL_JUMP_THRESHOLD = 32.0
RECTIFIED_CHECK_LINE_COUNT = 8


@dataclass(frozen=True)
class StripeConfig:
    name: str
    slug: str
    phase_steps: int
    cycles: tuple[int, ...]
    mode: str
    do_reconstruct: bool = True
    projector_width: int = 1280
    wrapped_steps: tuple[int, ...] = ()


STRIPE_CONFIGS = {
    0: StripeConfig("three_frequency_twelve_step", "three_frequency_12step", 12, (70, 75, 80), "multi_frequency"),
    1: StripeConfig("three_frequency_six_step", "three_frequency_6step", 6, (70, 75, 80), "multi_frequency"),
    2: StripeConfig("three_frequency_four_step", "three_frequency_4step", 4, (70, 75, 80), "multi_frequency"),
    3: StripeConfig("three_frequency_three_step", "three_frequency_3step", 3, (70, 75, 80), "multi_frequency"),
    4: StripeConfig("complementary_gray_code", "complementary_gray_code", 4, (16,), "gray_code"),
    5: StripeConfig("dual_frequency_custom", "dual_frequency_custom", 0, (80, 160, 160), "wrapped_only", False, wrapped_steps=(4, 4, 6)),
}


def reconstruct_capture(
    capture_dir: Union[Path, str],
    stripe_index: int,
    calib_path: Union[Path, str] = DEFAULT_CALIB_PATH,
) -> tuple[Path, dict[str, float]]:
    config = STRIPE_CONFIGS[int(stripe_index)]
    capture_path = Path(capture_dir)
    output_dir = capture_path / "rebuild"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_rebuild_outputs(output_dir)

    timing: dict[str, float] = {}

    decode_start = time.monotonic()
    left_images = load_capture_images(capture_path / LEFT_DIR_NAME)
    right_images = load_capture_images(capture_path / RIGHT_DIR_NAME)
    white_left, pattern_left = split_white_and_patterns(left_images, expected_pattern_count(config))
    white_right, pattern_right = split_white_and_patterns(right_images, expected_pattern_count(config))

    left_result = decode_patterns(pattern_left, config, white_left, output_dir / "left")
    right_result = decode_patterns(pattern_right, config, white_right, output_dir / "right")
    timing["phase_decode_ms"] = (time.monotonic() - decode_start) * 1000.0

    if config.do_reconstruct:
        reconstruct_start = time.monotonic()
        timing.update(build_point_cloud(
            calib_path=Path(calib_path),
            output_dir=output_dir,
        ))
        timing["reconstruct_ms"] = (time.monotonic() - reconstruct_start) * 1000.0
    else:
        timing["absolute_phase_ms"] = 0.0
        timing["reconstruct_ms"] = 0.0
        timing["point_cloud_ms"] = 0.0

    return output_dir, timing


def expected_pattern_count(config: StripeConfig) -> int:
    if config.mode == "multi_frequency":
        return config.phase_steps * len(config.cycles)
    if config.mode == "gray_code":
        return 9
    if config.mode == "wrapped_only":
        return sum(wrapped_only_steps(config))
    raise ValueError("Unsupported stripe mode: %s" % config.mode)


def wrapped_only_steps(config: StripeConfig) -> tuple[int, ...]:
    if config.wrapped_steps:
        return config.wrapped_steps
    return tuple([config.phase_steps] * len(config.cycles))


def cleanup_rebuild_outputs(output_dir: Path) -> None:
    patterns = [
        "disparity.png",
        "point_cloud.ply",
        "point_cloud_points.npy",
        "point_cloud_filter_stats.txt",
        "rectified_phase_pair.png",
        "rectified_valid_mask.png",
        "rectified_valid_stats.txt",
    ]
    side_patterns = [
        "wrapped_phase_*.npy",
        "wrapped_phase_*.png",
        "modulation_*.png",
        "projector_coord.npy",
        "gray_period_index.png",
    ]
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for side in ("left", "right"):
        side_dir = output_dir / side
        if not side_dir.exists():
            continue
        for pattern in side_patterns:
            for path in side_dir.glob(pattern):
                if path.is_file():
                    path.unlink()


def load_capture_images(image_dir: Path) -> list[np.ndarray]:
    if not image_dir.exists():
        raise FileNotFoundError("Image directory does not exist: %s" % image_dir)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ],
        key=capture_sort_key,
    )
    if not image_paths:
        raise RuntimeError("No images found in %s" % image_dir)

    images: list[np.ndarray] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("Failed to read image: %s" % path)
        images.append(image.astype(np.float32))
    return images


def split_white_and_patterns(
    images: Sequence[np.ndarray],
    pattern_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    needed = int(pattern_count) + 1
    if len(images) < needed:
        raise RuntimeError("Need at least %d captured frames, got %d" % (needed, len(images)))
    return images[0], list(images[1:needed])


def capture_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)_", path.name)
    if match is not None:
        return (0, int(match.group(1)), path.name)
    return (1, path.stat().st_mtime_ns, path.name)


def decode_patterns(
    images: Sequence[np.ndarray],
    config: StripeConfig,
    white_image: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.mode == "multi_frequency":
        return decode_multi_frequency(images, config, white_image, output_dir)
    if config.mode == "gray_code":
        return decode_gray_code(images, config, white_image, output_dir)
    if config.mode == "wrapped_only":
        return decode_wrapped_only(images, config, white_image, output_dir)
    raise ValueError("Unsupported stripe mode: %s" % config.mode)


def decode_multi_frequency(
    images: Sequence[np.ndarray],
    config: StripeConfig,
    white_image: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    groups = [
        images[index * config.phase_steps : (index + 1) * config.phase_steps]
        for index in range(len(config.cycles))
    ]
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        results = list(executor.map(_phase_shift_group, [(group, white_image) for group in groups]))

    wrapped_maps: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    for cycles, (wrapped, modulation, valid) in zip(config.cycles, results):
        wrapped_maps.append(wrapped)
        valid_maps.append(valid)
        save_phase_npy(output_dir / ("wrapped_phase_f%d" % cycles), wrapped, valid)

    valid = np.logical_and.reduce(valid_maps)
    projector_coord = unwrap_multi_frequency(
        wrapped_maps,
        config.cycles,
        valid,
        config.projector_width,
    )
    unwrapped_phase = (projector_coord / float(config.cycles[0])) * TWO_PI
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    return {
        "wrapped": wrapped_maps[0],
        "unwrapped": unwrapped_phase,
        "projector_coord": projector_coord,
        "valid": valid,
    }


def decode_three_step_frequency(
    images: Sequence[np.ndarray],
    config: StripeConfig,
    white_image: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    groups = [
        images[index * config.phase_steps : (index + 1) * config.phase_steps]
        for index in range(len(config.cycles))
    ]
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        results = list(executor.map(_phase_shift_group, [(group, white_image) for group in groups]))

    wrapped_maps: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    for period, (wrapped, modulation, valid) in zip(config.cycles, results):
        wrapped_maps.append(wrapped)
        valid_maps.append(valid)
        save_phase_npy(output_dir / ("wrapped_phase_f%d" % period), wrapped, valid)

    valid = np.logical_and.reduce(valid_maps)
    projector_coord = unwrap_multi_frequency(
        wrapped_maps,
        config.cycles,
        valid,
        config.projector_width,
    )
    unwrapped_phase = (projector_coord / float(config.cycles[0])) * TWO_PI
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    return {
        "wrapped": wrapped_maps[0],
        "unwrapped": unwrapped_phase,
        "projector_coord": projector_coord,
        "valid": valid,
    }


def decode_gray_code(
    images: Sequence[np.ndarray],
    config: StripeConfig,
    white_image: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    phase_images = images[:4]
    gray_images = images[4:8]
    complementary = images[8]
    wrapped, modulation, phase_valid = phase_shift(phase_images, white_image)
    period_count = config.cycles[0]

    threshold = np.maximum(white_image * 0.5, 8.0)
    bits = [(gray > threshold).astype(np.uint8) for gray in gray_images]
    gray_value = np.zeros_like(bits[0], dtype=np.uint8)
    for bit in bits:
        gray_value = (gray_value << 1) | bit
    comp_bit = (complementary > threshold).astype(np.uint8)
    period_index = decode_complementary_gray_order(
        gray_value,
        wrapped,
        comp_bit,
        period_count,
    )

    phase_period = float(config.projector_width) / float(period_count)
    unwrapped_phase = wrapped + period_index.astype(np.float32) * TWO_PI
    gray_valid = (
        (period_index >= 0)
        & (period_index < int(period_count))
        & np.isfinite(unwrapped_phase)
        & (unwrapped_phase >= 0.0)
        & (unwrapped_phase < float(period_count) * TWO_PI)
    )
    valid = phase_valid & gray_valid

    projector_coord = (unwrapped_phase / TWO_PI) * phase_period
    projector_coord[~valid] = np.nan
    unwrapped_phase[~valid] = np.nan

    save_phase_npy(output_dir / "wrapped_phase_f16", wrapped, phase_valid)
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    return {
        "wrapped": wrapped,
        "unwrapped": unwrapped_phase,
        "projector_coord": projector_coord,
        "valid": valid,
    }


def decode_wrapped_only(
    images: Sequence[np.ndarray],
    config: StripeConfig,
    white_image: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    first_result: Optional[dict[str, np.ndarray]] = None
    duplicate_cycles = len(set(config.cycles)) != len(config.cycles)
    start = 0
    for cycles, steps in zip(config.cycles, wrapped_only_steps(config)):
        group = images[start : start + steps]
        start += steps
        if len(group) != steps:
            raise RuntimeError("Need %d frames for wrapped phase P%d, got %d" % (steps, cycles, len(group)))
        wrapped, modulation, valid = phase_shift(group, white_image)
        base_name = "wrapped_phase_f%d_step%d" % (cycles, steps) if duplicate_cycles else "wrapped_phase_f%d" % cycles
        save_phase(output_dir / base_name, wrapped, valid)
        save_phase_npy(output_dir / base_name, wrapped, valid)
        save_wrapped_phase_binary_outputs(
            output_dir / base_name,
            wrapped,
            valid,
        )
        if first_result is None:
            projector_coord = (wrapped / TWO_PI) * float(cycles)
            projector_coord[~valid] = np.nan
            first_result = {
                "wrapped": wrapped,
                "unwrapped": wrapped,
                "projector_coord": projector_coord,
                "valid": valid,
            }
    if first_result is None:
        raise RuntimeError("No wrapped phase was decoded.")
    return first_result


def phase_shift(
    images: Sequence[np.ndarray],
    white_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(images, axis=0).astype(np.float32)
    steps = stack.shape[0]
    if steps == 3:
        return phase_shift_three_step_images(stack[0], stack[1], stack[2], white_image)

    phase_steps = np.arange(steps, dtype=np.float32) * (TWO_PI / float(steps))
    cos_terms = np.cos(phase_steps)[:, None, None]
    sin_terms = np.sin(phase_steps)[:, None, None]
    real = np.sum(stack * cos_terms, axis=0)
    imag = -np.sum(stack * sin_terms, axis=0)
    wrapped = np.mod(np.arctan2(imag, real), TWO_PI).astype(np.float32)
    modulation = (2.0 / float(steps)) * np.sqrt(real * real + imag * imag)
    valid = build_valid_mask(stack, white_image, modulation)
    return wrapped, modulation.astype(np.float32), valid


def _phase_shift_group(args: tuple[Sequence[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group, white_image = args
    return phase_shift(group, white_image)


def phase_shift_three_step_images(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    white_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = np.sqrt(3.0) * (third - second)
    denominator = 2.0 * first - second - third
    wrapped = np.mod(np.arctan2(numerator, denominator), TWO_PI).astype(np.float32)
    modulation = np.sqrt(numerator * numerator + denominator * denominator) / 3.0
    stack = np.stack((first, second, third), axis=0)
    valid = build_valid_mask(stack, white_image, modulation)
    return wrapped, modulation.astype(np.float32), valid


def build_valid_mask(
    stack: np.ndarray,
    white_image: np.ndarray,
    modulation: np.ndarray,
) -> np.ndarray:
    average = np.mean(stack, axis=0)
    intensity_swing = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    white_threshold = max(8.0, float(np.nanpercentile(white_image, 10)) * 0.25)
    modulation_threshold = max(
        MIN_PHASE_MODULATION,
        float(np.nanpercentile(modulation, 95)) * PHASE_MODULATION_RELATIVE_THRESHOLD,
    )
    swing_threshold = max(
        MIN_PHASE_INTENSITY_SWING,
        float(np.nanpercentile(intensity_swing, 95)) * PHASE_SWING_RELATIVE_THRESHOLD,
    )
    return (
        np.isfinite(average)
        & np.isfinite(modulation)
        & np.isfinite(intensity_swing)
        & (white_image > white_threshold)
        & (modulation > modulation_threshold)
        & (intensity_swing > swing_threshold)
    )


def unwrap_multi_frequency(
    wrapped_maps: Sequence[np.ndarray],
    periods: Sequence[int],
    valid: np.ndarray,
    projector_width: int,
) -> np.ndarray:
    if len(wrapped_maps) == 3 and len(periods) == 3:
        residue0 = (np.mod(wrapped_maps[0], TWO_PI) / TWO_PI) * float(periods[0])
        residue1 = (np.mod(wrapped_maps[1], TWO_PI) / TWO_PI) * float(periods[1])
        residue2 = (np.mod(wrapped_maps[2], TWO_PI) / TWO_PI) * float(periods[2])
        max_order = int(np.ceil(float(projector_width) / float(periods[0]))) + 2
        return unwrap_multi_frequency_parallel(
            residue0.astype(np.float32),
            residue1.astype(np.float32),
            residue2.astype(np.float32),
            valid.astype(np.bool_),
            np.int32(max_order),
            np.float32(periods[0]),
            np.float32(periods[1]),
            np.float32(periods[2]),
            np.float32(projector_width),
        )

    residues = [
        (np.mod(phase, TWO_PI) / TWO_PI) * float(period)
        for phase, period in zip(wrapped_maps, periods)
    ]
    base_period = int(periods[0])
    max_coord = float(projector_width)
    max_order = int(np.ceil(max_coord / float(base_period))) + 2
    best_score = np.full(residues[0].shape, np.inf, dtype=np.float32)
    best_coord = np.zeros(residues[0].shape, dtype=np.float32)

    for order in range(max_order):
        candidate = residues[0] + float(order * base_period)
        score = np.zeros(candidate.shape, dtype=np.float32)
        for residue, period in zip(residues[1:], periods[1:]):
            error = circular_period_distance(candidate, residue, float(period))
            score += error * error
        in_range = (candidate >= 0.0) & (candidate < max_coord)
        better = valid & in_range & (score < best_score)
        best_score[better] = score[better]
        best_coord[better] = candidate[better]

    best_coord[~valid] = np.nan
    return best_coord.astype(np.float32)


def circular_period_distance(
    coordinate: np.ndarray,
    residue: np.ndarray,
    period: float,
) -> np.ndarray:
    return np.abs(np.mod(coordinate - residue + period * 0.5, period) - period * 0.5).astype(np.float32)


def gray_to_binary(gray: np.ndarray) -> np.ndarray:
    binary = gray.copy()
    shift = 1
    while shift < 8:
        binary ^= binary >> shift
        shift <<= 1
    return binary


def decode_complementary_gray_order(
    gray_value: np.ndarray,
    wrapped_phase: np.ndarray,
    comp_bit: np.ndarray,
    period_count: int,
) -> np.ndarray:
    switch_phase = wrap_to_pi(wrapped_phase - np.pi)
    base_order = gray_to_binary(gray_value).astype(np.int16)
    k1 = np.clip(base_order, 0, int(period_count) - 1)

    k2_same = np.clip(base_order, 0, int(period_count))
    k2_next = np.clip(base_order + 1, 0, int(period_count))
    k2 = np.where((k2_same & 1) == comp_bit.astype(np.int16), k2_same, k2_next)

    order = np.empty(base_order.shape, dtype=np.int16)
    first_region = switch_phase < (-0.5 * np.pi)
    middle_region = (switch_phase >= (-0.5 * np.pi)) & (switch_phase <= (0.5 * np.pi))
    order[first_region] = k2[first_region]
    order[middle_region] = k1[middle_region]
    order[~(first_region | middle_region)] = k2[~(first_region | middle_region)] - 1
    return np.clip(order, 0, int(period_count) - 1).astype(np.int16)


def wrap_to_pi(phase: np.ndarray) -> np.ndarray:
    return (np.mod(phase + np.pi, TWO_PI) - np.pi).astype(np.float32)


@njit(parallel=True, cache=True, fastmath=True)
def unwrap_multi_frequency_parallel(
    residue0: np.ndarray,
    residue1: np.ndarray,
    residue2: np.ndarray,
    valid: np.ndarray,
    max_order: np.int32,
    period0: np.float32,
    period1: np.float32,
    period2: np.float32,
    projector_width: np.float32,
) -> np.ndarray:
    height, width = residue0.shape
    best_coord = np.empty((height, width), dtype=np.float32)
    for row in prange(height):
        for col in range(width):
            if not valid[row, col]:
                best_coord[row, col] = np.nan
                continue

            base = residue0[row, col]
            best_score = np.float32(np.inf)
            best_value = np.float32(np.nan)

            for order in range(max_order):
                candidate = base + np.float32(order) * period0
                if candidate < 0.0 or candidate >= projector_width:
                    continue

                err1 = circular_period_distance_scalar(candidate, residue1[row, col], period1)
                err2 = circular_period_distance_scalar(candidate, residue2[row, col], period2)
                score = err1 * err1 + err2 * err2
                if score < best_score:
                    best_score = score
                    best_value = candidate

            best_coord[row, col] = best_value

    return best_coord


@njit(cache=True, fastmath=True)
def circular_period_distance_scalar(coordinate: np.float32, residue: np.float32, period: np.float32) -> np.float32:
    half_period = np.float32(0.5) * period
    return np.float32(np.abs(((coordinate - residue + half_period) % period) - half_period))


@njit(parallel=True, cache=True, fastmath=True)
def match_phase_to_disparity_parallel(
    left_coord: np.ndarray,
    right_coord: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> np.ndarray:
    height, width = left_coord.shape
    disparity = np.empty((height, width), dtype=np.float32)
    x_grid = np.arange(width, dtype=np.float32)

    for row in prange(height):
        for col in range(width):
            disparity[row, col] = np.nan

        right_count = 0
        left_count = 0
        for col in range(width):
            if right_valid[row, col] and np.isfinite(right_coord[row, col]):
                right_count += 1
            if left_valid[row, col] and np.isfinite(left_coord[row, col]):
                left_count += 1

        if right_count < 2 or left_count == 0:
            continue

        right_phase = np.empty(right_count, dtype=np.float32)
        right_x = np.empty(right_count, dtype=np.float32)
        idx = 0
        for col in range(width):
            if right_valid[row, col] and np.isfinite(right_coord[row, col]):
                right_phase[idx] = right_coord[row, col]
                right_x[idx] = x_grid[col]
                idx += 1

        order = np.argsort(right_phase)
        sorted_phase = np.empty(right_count, dtype=np.float32)
        sorted_x = np.empty(right_count, dtype=np.float32)
        for i in range(right_count):
            sorted_phase[i] = right_phase[order[i]]
            sorted_x[i] = right_x[order[i]]

        unique_phase = np.empty(right_count, dtype=np.float32)
        unique_x = np.empty(right_count, dtype=np.float32)
        unique_count = 0
        unique_phase[0] = sorted_phase[0]
        unique_x[0] = sorted_x[0]
        unique_count = 1
        for i in range(1, right_count):
            if sorted_phase[i] != unique_phase[unique_count - 1]:
                unique_phase[unique_count] = sorted_phase[i]
                unique_x[unique_count] = sorted_x[i]
                unique_count += 1

        if unique_count < 2:
            continue

        min_phase = unique_phase[0]
        max_phase = unique_phase[unique_count - 1]

        for col in range(width):
            if not (left_valid[row, col] and np.isfinite(left_coord[row, col])):
                continue

            phase = left_coord[row, col]
            if phase < min_phase or phase > max_phase:
                continue

            insert_idx = phase_binary_search(unique_phase, unique_count, phase)
            if insert_idx <= 0:
                insert_idx = 1
            elif insert_idx >= unique_count:
                insert_idx = unique_count - 1

            phase_gap = unique_phase[insert_idx] - unique_phase[insert_idx - 1]
            x_gap = np.abs(unique_x[insert_idx] - unique_x[insert_idx - 1])
            if phase_gap > PHASE_JUMP_THRESHOLD or x_gap > PIXEL_JUMP_THRESHOLD:
                continue

            p0 = unique_phase[insert_idx - 1]
            p1 = unique_phase[insert_idx]
            x0 = unique_x[insert_idx - 1]
            x1 = unique_x[insert_idx]
            if p1 == p0:
                matched_x = x0
            else:
                matched_x = x0 + (phase - p0) * (x1 - x0) / (p1 - p0)
            disparity[row, col] = x_grid[col] - matched_x

    return disparity


@njit(cache=True, fastmath=True)
def phase_binary_search(values: np.ndarray, count: int, target: float) -> int:
    lo = 0
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True, cache=True, fastmath=True)
def calculate_xyz_parallel(
    disparity: np.ndarray,
    Q: np.ndarray,
    min_depth: np.float32,
    max_depth: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = disparity.shape
    points = np.empty((height, width, 3), dtype=np.float32)
    base_valid = np.zeros((height, width), dtype=np.bool_)

    q03 = np.float32(Q[0, 3])
    q13 = np.float32(Q[1, 3])
    q23 = np.float32(Q[2, 3])
    q32 = np.float32(Q[3, 2])
    q33 = np.float32(Q[3, 3])

    for row in prange(height):
        for col in range(width):
            d = disparity[row, col]
            if not np.isfinite(d) or np.abs(d) <= np.float32(1e-6):
                points[row, col, 0] = np.nan
                points[row, col, 1] = np.nan
                points[row, col, 2] = np.nan
                continue

            w0 = q32 * d + q33
            if np.abs(w0) <= np.float32(1e-12):
                points[row, col, 0] = np.nan
                points[row, col, 1] = np.nan
                points[row, col, 2] = np.nan
                continue

            x = (np.float32(col) + q03) / w0
            y = (np.float32(row) + q13) / w0
            z = q23 / w0
            points[row, col, 0] = x
            points[row, col, 1] = y
            points[row, col, 2] = z
            if min_depth < z < max_depth:
                base_valid[row, col] = True

    return points, base_valid


def build_point_cloud(
    calib_path: Path,
    output_dir: Path,
) -> dict[str, float]:
    timing: dict[str, float] = {}
    phase_to_abs_start = time.monotonic()
    left_phase, left_valid = load_unwrapped_phase_npy(output_dir / "left" / "unwrapped_phase.npy")
    right_phase, right_valid = load_unwrapped_phase_npy(output_dir / "right" / "unwrapped_phase.npy")
    if left_phase.shape != right_phase.shape:
        raise RuntimeError(
            "Left/right unwrapped phase sizes differ: %s vs %s"
            % (left_phase.shape, right_phase.shape)
        )

    calib = load_calibration(calib_path)
    height, width = left_phase.shape
    rectified = rectify_phase_pair(
        left_phase,
        right_phase,
        left_valid,
        right_valid,
        calib,
        (width, height),
    )
    timing["absolute_phase_ms"] = (time.monotonic() - phase_to_abs_start) * 1000.0

    reconstruct_start = time.monotonic()
    disparity = match_phase_to_disparity_parallel(
        rectified["left_coord"],
        rectified["right_coord"],
        rectified["left_valid"],
        rectified["right_valid"],
    )
    np.save(str(output_dir / "disparity.npy"), disparity.astype(np.float32))
    points, base_valid = calculate_xyz_parallel(
        disparity.astype(np.float32),
        rectified["Q"].astype(np.float64),
        np.float32(POINT_CLOUD_Z_MIN_MM),
        np.float32(POINT_CLOUD_Z_MAX_MM),
    )
    valid = filter_point_cloud_valid_mask(points, base_valid)
    save_point_cloud_txt(output_dir / "point_cloud.txt", points, valid)
    timing["point_cloud_ms"] = (time.monotonic() - reconstruct_start) * 1000.0
    return timing


def load_unwrapped_phase_npy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError("Unwrapped phase data does not exist: %s" % path)
    phase = np.load(str(path)).astype(np.float32)
    valid = np.isfinite(phase)
    if phase.ndim != 2:
        raise RuntimeError("Unwrapped phase data must be a 2D array: %s" % path)
    if not np.any(valid):
        raise RuntimeError("Unwrapped phase data has no valid pixels: %s" % path)
    return phase, valid


def filter_point_cloud_valid_mask(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    z = points[:, :, 2]
    filtered = (
        valid
        & np.isfinite(z)
        & (z >= POINT_CLOUD_Z_MIN_MM)
        & (z <= POINT_CLOUD_Z_MAX_MM)
    )
    if POINT_CLOUD_MIN_COMPONENT_PIXELS <= 1 or not np.any(filtered):
        return filtered

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        filtered.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return filtered

    areas = stats[:, cv2.CC_STAT_AREA]
    keep_labels = np.flatnonzero(areas >= int(POINT_CLOUD_MIN_COMPONENT_PIXELS))
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(filtered, dtype=bool)
    return np.isin(labels, keep_labels)


def load_calibration(calib_path: Path) -> dict[str, np.ndarray]:
    text = calib_path.read_text(encoding="utf-8")
    return {
        "image_size": read_named_matrix(text, "imageSize").reshape(-1),
        "K1": matlab_intrinsic_to_opencv(read_named_matrix(text, "KK_L")),
        "K2": matlab_intrinsic_to_opencv(read_named_matrix(text, "KK_R")),
        "dist1": distortion_vector(
            read_named_matrix(text, "RadialDistortion_L"),
            read_named_matrix(text, "TangentialDistortion_L"),
        ),
        "dist2": distortion_vector(
            read_named_matrix(text, "RadialDistortion_R"),
            read_named_matrix(text, "TangentialDistortion_R"),
        ),
        "R": read_named_matrix(text, "R").T.copy(),
        "T": read_named_matrix(text, "T").T.reshape(3, 1),
    }


def read_named_matrix(text: str, name: str) -> np.ndarray:
    match = re.search(r"(?m)^%s\s*\n(.*?)(?=\n[A-Za-z_][A-Za-z0-9_]*\s*\n|\Z)" % re.escape(name), text, re.S)
    if match is None:
        raise RuntimeError("Calibration field not found: %s" % name)
    rows = []
    for line in match.group(1).splitlines():
        values = [float(item) for item in line.split()]
        if values:
            rows.append(values)
    if not rows:
        raise RuntimeError("Calibration field is empty: %s" % name)
    return np.array(rows, dtype=np.float64)


def matlab_intrinsic_to_opencv(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (3, 3):
        raise RuntimeError("Intrinsic matrix must be 3x3.")
    return matrix.T.copy()


def distortion_vector(radial: np.ndarray, tangential: np.ndarray) -> np.ndarray:
    radial = radial.reshape(-1)
    tangential = tangential.reshape(-1)
    return np.array(
        [radial[0], radial[1], tangential[0], tangential[1], 0.0],
        dtype=np.float64,
    )


def rectify_phase_pair(
    left_phase: np.ndarray,
    right_phase: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    calib: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    calib = scale_calibration_to_image_size(calib, image_size)
    R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
        calib["K1"],
        calib["dist1"],
        calib["K2"],
        calib["dist2"],
        image_size,
        calib["R"],
        calib["T"],
        flags=0,
        alpha=-1,
        newImageSize=image_size,
    )
    Q = left_rectification_to_world_matrix(R1) @ Q
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"],
        calib["dist1"],
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["K2"],
        calib["dist2"],
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )
    rectified_left_valid = remap_mask(left_valid, map1x, map1y)
    rectified_right_valid = remap_mask(right_valid, map2x, map2y)
    rectified_left_phase = remap_float(left_phase, map1x, map1y)
    rectified_right_phase = remap_float(right_phase, map2x, map2y)
    rectified_left_phase[~rectified_left_valid] = np.nan
    rectified_right_phase[~rectified_right_valid] = np.nan
    return {
        "left_coord": rectified_left_phase,
        "right_coord": rectified_right_phase,
        "left_valid": rectified_left_valid,
        "right_valid": rectified_right_valid,
        "Q": Q,
    }


def left_rectification_to_world_matrix(rectification: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.linalg.inv(rectification)
    return transform


def scale_calibration_to_image_size(
    calib: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    raw_size = calib.get("image_size")
    if raw_size is None or raw_size.size < 2:
        return calib

    raw0 = float(raw_size[0])
    raw1 = float(raw_size[1])
    image_width, image_height = image_size
    if abs(raw0 - image_height) < abs(raw0 - image_width):
        calib_width, calib_height = raw1, raw0
    else:
        calib_width, calib_height = raw0, raw1

    if calib_width <= 0 or calib_height <= 0:
        return calib

    sx = float(image_width) / float(calib_width)
    sy = float(image_height) / float(calib_height)
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return calib

    scaled = dict(calib)
    for key in ("K1", "K2"):
        matrix = np.array(calib[key], dtype=np.float64, copy=True)
        matrix[0, 0] *= sx
        matrix[0, 2] *= sx
        matrix[1, 1] *= sy
        matrix[1, 2] *= sy
        scaled[key] = matrix
    return scaled


def remap_float(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(image), image, -1.0).astype(np.float32)
    remapped = cv2.remap(filled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    remapped[remapped < 0.0] = np.nan
    return remapped


def remap_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def save_timing_stats(path: Path, timing: dict[str, float]) -> None:
    point_cloud_ms = float(timing.get("point_cloud_ms", timing.get("reconstruct_ms", 0.0)))
    lines = [
        "timing1_capture_to_save_ms=%.3f" % float(timing.get("capture_save_ms", 0.0)),
        "timing2_phase_decode_ms=%.3f" % float(timing.get("phase_decode_ms", 0.0)),
        "timing3_point_cloud_ms=%.3f" % point_cloud_ms,
        "timing4_total_ms=%.3f" % float(timing.get("total_ms", 0.0)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_phase_npy(
    base_path: Path,
    phase: np.ndarray,
    valid: np.ndarray,
) -> None:
    masked_phase = phase.astype(np.float32, copy=True)
    masked_phase[~valid] = np.nan
    np.save(str(base_path.with_suffix(".npy")), masked_phase)


def save_phase(
    base_path: Path,
    phase: np.ndarray,
    valid: np.ndarray,
) -> None:
    masked_phase = phase.astype(np.float32, copy=True)
    masked_phase[~valid] = np.nan
    np.save(str(base_path.with_suffix(".npy")), masked_phase)

    finite = np.isfinite(masked_phase)
    image = np.zeros(masked_phase.shape, dtype=np.uint8)
    if np.any(finite):
        lo = float(np.nanpercentile(masked_phase[finite], 1))
        hi = float(np.nanpercentile(masked_phase[finite], 99))
        if hi <= lo:
            hi = lo + 1.0
        scaled = (masked_phase - lo) * (255.0 / (hi - lo))
        image[finite] = np.clip(scaled[finite], 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(base_path.with_suffix(".png")), image)
    if not ok:
        raise RuntimeError("Failed to save phase image: %s" % base_path.with_suffix(".png"))


def save_wrapped_phase_binary_outputs(
    base_path: Path,
    phase: np.ndarray,
    valid: np.ndarray,
) -> None:
    phase_mod = np.mod(phase, TWO_PI)
    finite = np.isfinite(phase_mod) & valid

    binary_1 = np.full(phase_mod.shape, np.nan, dtype=np.float32)
    mask_1 = finite & (phase_mod >= (np.pi / 4.0)) & (phase_mod <= (7.0 * np.pi / 4.0))
    binary_1[finite] = 0.0
    binary_1[mask_1] = 255.0
    save_binary_phase(base_path.with_name(base_path.name + "_binary_1"), binary_1, finite)

    binary_2 = np.full(phase_mod.shape, np.nan, dtype=np.float32)
    mask_2 = finite & (
        ((phase_mod > 0.0) & (phase_mod <= (3.0 * np.pi / 4.0)))
        | ((phase_mod >= (5.0 * np.pi / 4.0)) & (phase_mod <= TWO_PI))
    )
    binary_2[finite] = 0.0
    binary_2[mask_2] = 255.0
    save_binary_phase(base_path.with_name(base_path.name + "_binary_2"), binary_2, finite)


def save_binary_phase(
    base_path: Path,
    binary: np.ndarray,
    valid: np.ndarray,
) -> None:
    np.save(str(base_path.with_suffix(".npy")), binary.astype(np.float32))
    image = np.zeros(binary.shape, dtype=np.uint8)
    finite = np.isfinite(binary) & valid
    image[finite] = np.clip(binary[finite], 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(base_path.with_suffix(".png")), image)
    if not ok:
        raise RuntimeError("Failed to save binary phase image: %s" % base_path.with_suffix(".png"))


def save_point_cloud_txt(path: Path, points: np.ndarray, valid: np.ndarray) -> None:
    flat_points = points[valid].astype(np.float32)
    np.savetxt(str(path), flat_points, fmt="%.8f %.8f %.8f")
