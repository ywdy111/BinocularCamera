from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIB_PATH = PROJECT_ROOT / "config" / "data_calib" / "calib.txt"
LEFT_DIR_NAME = "Left_picture"
RIGHT_DIR_NAME = "Right_picture"
TWO_PI = 2.0 * np.pi
POINT_CLOUD_Z_MIN_MM = 300.0
POINT_CLOUD_Z_MAX_MM = 500.0
POINT_CLOUD_MIN_COMPONENT_PIXELS = 64


@dataclass(frozen=True)
class StripeConfig:
    name: str
    slug: str
    phase_steps: int
    cycles: tuple[int, ...]
    mode: str
    do_reconstruct: bool = True
    projector_width: int = 1280


STRIPE_CONFIGS = {
    0: StripeConfig("three_frequency_twelve_step", "three_frequency_12step", 12, (70, 75, 80), "multi_frequency"),
    1: StripeConfig("three_frequency_six_step", "three_frequency_6step", 6, (70, 75, 80), "multi_frequency"),
    2: StripeConfig("three_frequency_four_step", "three_frequency_4step", 4, (70, 75, 80), "multi_frequency"),
    3: StripeConfig("three_frequency_three_step", "three_frequency_3step", 3, (70, 75, 80), "multi_frequency"),
    4: StripeConfig("complementary_gray_code", "complementary_gray_code", 4, (16,), "gray_code"),
    5: StripeConfig("dual_frequency_custom", "dual_frequency_custom", 4, (80, 40), "wrapped_only", False),
}


def reconstruct_capture(
    capture_dir: Union[Path, str],
    stripe_index: int,
    calib_path: Union[Path, str] = DEFAULT_CALIB_PATH,
) -> Path:
    config = STRIPE_CONFIGS[int(stripe_index)]
    capture_path = Path(capture_dir)
    output_dir = capture_path / "rebuild"
    output_dir.mkdir(parents=True, exist_ok=True)

    left_images = load_capture_images(capture_path / LEFT_DIR_NAME)
    right_images = load_capture_images(capture_path / RIGHT_DIR_NAME)
    white_left, pattern_left = split_white_and_patterns(left_images, expected_pattern_count(config))
    white_right, pattern_right = split_white_and_patterns(right_images, expected_pattern_count(config))

    left_result = decode_patterns(pattern_left, config, white_left, output_dir / "left")
    right_result = decode_patterns(pattern_right, config, white_right, output_dir / "right")

    if config.do_reconstruct:
        build_point_cloud(
            calib_path=Path(calib_path),
            output_dir=output_dir,
        )

    return output_dir


def expected_pattern_count(config: StripeConfig) -> int:
    if config.mode == "multi_frequency":
        return config.phase_steps * len(config.cycles)
    if config.mode == "gray_code":
        return 9
    if config.mode == "wrapped_only":
        return config.phase_steps * len(config.cycles)
    raise ValueError("Unsupported stripe mode: %s" % config.mode)


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
    wrapped_maps: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    for index, cycles in enumerate(config.cycles):
        start = index * config.phase_steps
        group = images[start : start + config.phase_steps]
        wrapped, modulation, valid = phase_shift(group, white_image)
        wrapped_maps.append(wrapped)
        valid_maps.append(valid)
        save_phase(output_dir / ("wrapped_phase_f%d" % cycles), wrapped, valid, period=TWO_PI)
        save_float_image(output_dir / ("modulation_f%d.png" % cycles), modulation, valid)

    valid = np.logical_and.reduce(valid_maps)
    projector_coord = unwrap_multi_frequency(
        wrapped_maps,
        config.cycles,
        valid,
        config.projector_width,
    )
    unwrapped_phase = (projector_coord / float(config.cycles[0])) * TWO_PI
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    np.save(str(output_dir / "projector_coord.npy"), projector_coord.astype(np.float32))
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
    wrapped_maps: list[np.ndarray] = []
    valid_maps: list[np.ndarray] = []
    for index, period in enumerate(config.cycles):
        start = index * config.phase_steps
        first, second, third = images[start : start + config.phase_steps]
        wrapped, modulation, valid = phase_shift_three_step_images(
            first,
            second,
            third,
            white_image,
        )
        wrapped_maps.append(wrapped)
        valid_maps.append(valid)
        save_phase(output_dir / ("wrapped_phase_f%d" % period), wrapped, valid, period=TWO_PI)
        save_float_image(output_dir / ("modulation_f%d.png" % period), modulation, valid)

    valid = np.logical_and.reduce(valid_maps)
    projector_coord = unwrap_multi_frequency(
        wrapped_maps,
        config.cycles,
        valid,
        config.projector_width,
    )
    unwrapped_phase = (projector_coord / float(config.cycles[0])) * TWO_PI
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    np.save(str(output_dir / "projector_coord.npy"), projector_coord.astype(np.float32))
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
    period_index = gray_to_binary(gray_value)
    phase_fraction = wrapped / TWO_PI
    comp_bit = (complementary > threshold).astype(np.uint8)
    period_index = correct_with_complementary_gray(
        period_index,
        phase_fraction,
        comp_bit,
        period_count,
    )

    gray_valid = period_index < period_count
    valid = phase_valid & gray_valid

    phase_period = float(config.projector_width) / float(period_count)
    projector_coord = (period_index.astype(np.float32) + phase_fraction) * phase_period
    projector_coord[~valid] = np.nan
    unwrapped_phase = (period_index.astype(np.float32) + phase_fraction) * TWO_PI
    unwrapped_phase[~valid] = np.nan

    save_phase(output_dir / "wrapped_phase_f16", wrapped, phase_valid, period=TWO_PI)
    save_float_image(output_dir / "modulation_f16.png", modulation, phase_valid)
    save_phase(output_dir / "unwrapped_phase", unwrapped_phase, valid)
    save_float_image(output_dir / "gray_period_index.png", period_index.astype(np.float32), gray_valid)
    np.save(str(output_dir / "projector_coord.npy"), projector_coord.astype(np.float32))
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
    for index, cycles in enumerate(config.cycles):
        start = index * config.phase_steps
        group = images[start : start + config.phase_steps]
        wrapped, modulation, valid = phase_shift(group, white_image)
        save_phase(output_dir / ("wrapped_phase_f%d" % cycles), wrapped, valid, period=TWO_PI)
        save_float_image(output_dir / ("modulation_f%d.png" % cycles), modulation, valid)
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
    white_threshold = max(8.0, float(np.nanpercentile(white_image, 10)) * 0.25)
    modulation_threshold = max(3.0, float(np.nanpercentile(modulation, 75)) * 0.08)
    return (
        np.isfinite(average)
        & np.isfinite(modulation)
        & (white_image > white_threshold)
        & (modulation > modulation_threshold)
    )


def unwrap_multi_frequency(
    wrapped_maps: Sequence[np.ndarray],
    periods: Sequence[int],
    valid: np.ndarray,
    projector_width: int,
) -> np.ndarray:
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


def correct_with_complementary_gray(
    period_index: np.ndarray,
    phase_fraction: np.ndarray,
    comp_bit: np.ndarray,
    period_count: int,
) -> np.ndarray:
    shifted_index = period_index + (phase_fraction >= 0.5).astype(np.uint8)
    expected_comp = gray_lsb(shifted_index)
    mismatch = comp_bit != expected_comp

    corrected = period_index.astype(np.int16)
    corrected[mismatch & (phase_fraction < 0.5)] += 1
    corrected[mismatch & (phase_fraction >= 0.5)] -= 1
    corrected = np.clip(corrected, 0, int(period_count) - 1)
    return corrected.astype(np.uint8)


def gray_lsb(binary_value: np.ndarray) -> np.ndarray:
    gray = binary_value ^ (binary_value >> 1)
    return (gray & 1).astype(np.uint8)


def build_point_cloud(
    calib_path: Path,
    output_dir: Path,
) -> None:
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
    disparity = match_phase_to_disparity(
        rectified["left_coord"],
        rectified["right_coord"],
        rectified["left_valid"],
        rectified["right_valid"],
    )
    np.save(str(output_dir / "disparity.npy"), disparity.astype(np.float32))
    save_float_image(output_dir / "disparity.png", disparity, np.isfinite(disparity))

    points = cv2.reprojectImageTo3D(disparity.astype(np.float32), rectified["Q"])
    base_valid = np.isfinite(disparity) & np.isfinite(points).all(axis=2) & (np.abs(disparity) > 1e-6)
    valid = filter_point_cloud_valid_mask(points, base_valid)
    colors = normalize_to_uint8(rectified["left_coord"], rectified["left_valid"])
    save_ply(output_dir / "point_cloud.ply", points, valid, colors)
    save_point_cloud_txt(output_dir / "point_cloud.txt", points, valid)
    np.save(str(output_dir / "point_cloud_points.npy"), points[valid].astype(np.float32))
    save_point_cloud_filter_stats(output_dir / "point_cloud_filter_stats.txt", points, base_valid, valid)


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


def save_point_cloud_filter_stats(
    path: Path,
    points: np.ndarray,
    base_valid: np.ndarray,
    valid: np.ndarray,
) -> None:
    z = points[:, :, 2]
    z_range = base_valid & (z >= POINT_CLOUD_Z_MIN_MM) & (z <= POINT_CLOUD_Z_MAX_MM)
    lines = [
        "z_min_mm=%.3f" % POINT_CLOUD_Z_MIN_MM,
        "z_max_mm=%.3f" % POINT_CLOUD_Z_MAX_MM,
        "min_component_pixels=%d" % POINT_CLOUD_MIN_COMPONENT_PIXELS,
        "base_valid_points=%d" % int(np.count_nonzero(base_valid)),
        "z_range_points=%d" % int(np.count_nonzero(z_range)),
        "saved_points=%d" % int(np.count_nonzero(valid)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "R": read_named_matrix(text, "R"),
        "T": read_named_matrix(text, "T").reshape(3, 1),
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
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
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
    return {
        "left_coord": remap_float(left_phase, map1x, map1y),
        "right_coord": remap_float(right_phase, map2x, map2y),
        "left_valid": remap_mask(left_valid, map1x, map1y),
        "right_valid": remap_mask(right_valid, map2x, map2y),
        "Q": Q,
    }


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


def match_phase_to_disparity(
    left_coord: np.ndarray,
    right_coord: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> np.ndarray:
    height, width = left_coord.shape
    disparity = np.full((height, width), np.nan, dtype=np.float32)
    x_grid = np.arange(width, dtype=np.float32)

    for row in range(height):
        right_mask = right_valid[row] & np.isfinite(right_coord[row])
        left_mask = left_valid[row] & np.isfinite(left_coord[row])
        if np.count_nonzero(right_mask) < 2 or np.count_nonzero(left_mask) == 0:
            continue

        right_phase = right_coord[row, right_mask]
        right_x = x_grid[right_mask]
        order = np.argsort(right_phase)
        sorted_phase = right_phase[order]
        sorted_x = right_x[order]
        unique_phase, unique_indices = np.unique(sorted_phase, return_index=True)
        if unique_phase.size < 2:
            continue
        unique_x = sorted_x[unique_indices]

        left_phase = left_coord[row, left_mask]
        in_range = (left_phase >= unique_phase[0]) & (left_phase <= unique_phase[-1])
        if not np.any(in_range):
            continue
        left_indices = np.flatnonzero(left_mask)[in_range]
        matched_x = np.interp(left_phase[in_range], unique_phase, unique_x).astype(np.float32)
        disparity[row, left_indices] = x_grid[left_indices] - matched_x

    return disparity


def save_phase(
    base_path: Path,
    phase: np.ndarray,
    valid: np.ndarray,
    period: Optional[float] = None,
) -> None:
    masked_phase = phase.astype(np.float32, copy=True)
    masked_phase[~valid] = np.nan
    np.save(str(base_path.with_suffix(".npy")), masked_phase)
    finite = np.isfinite(masked_phase)
    if period is None:
        if np.any(finite):
            lo = float(np.nanpercentile(masked_phase[finite], 1))
            hi = float(np.nanpercentile(masked_phase[finite], 99))
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = 0.0, float(period)
    save_float_image(base_path.with_suffix(".png"), masked_phase, finite, lo=lo, hi=hi)


def save_float_image(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> None:
    image = normalize_to_uint8(values, valid, lo=lo, hi=hi)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError("Failed to save image: %s" % path)


def normalize_to_uint8(
    values: np.ndarray,
    valid: np.ndarray,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    finite = np.isfinite(values) & valid
    output = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(finite):
        return output
    if lo is None:
        lo = float(np.nanpercentile(values[finite], 1))
    if hi is None:
        hi = float(np.nanpercentile(values[finite], 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (values - lo) * (255.0 / (hi - lo))
    output[finite] = np.clip(scaled[finite], 0, 255).astype(np.uint8)
    return output


def save_point_cloud_txt(path: Path, points: np.ndarray, valid: np.ndarray) -> None:
    flat_points = points[valid].astype(np.float32)
    np.savetxt(str(path), flat_points, fmt="%.8f %.8f %.8f")


def save_ply(
    path: Path,
    points: np.ndarray,
    valid: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    flat_points = points[valid].astype(np.float32)
    if colors is None:
        flat_colors = np.full(flat_points.shape[0], 255, dtype=np.uint8)
    else:
        flat_colors = colors[valid].astype(np.uint8)

    vertex = np.empty(
        flat_points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = flat_points[:, 0]
    vertex["y"] = flat_points[:, 1]
    vertex["z"] = flat_points[:, 2]
    vertex["red"] = flat_colors
    vertex["green"] = flat_colors
    vertex["blue"] = flat_colors

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex %d\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ) % vertex.shape[0]
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertex.tofile(file)
