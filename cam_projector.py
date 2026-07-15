#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双目相机与投影仪联调测试。

投影仪播放一次烧录图案序列。在投影仪触发前打开左右两台相机，采集单色（黑白）图像，
并保存与投影序列相同数量的帧。预期捕获的第一张图像为全白帧，随后是条纹帧。
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from config.camera.CameraParams_header import MV_TRIGGER_SOURCE_LINE0
from config.camera.MvErrorDefine_const import MV_OK
from My_HKCamera import CameraError, My_HKCamera
from My_Projector import Projector
from real_camera import print_device_list

# 曝光时间，单位：微秒。根据现场亮度修改这个值即可。
EXPOSURE_TIME_US = 10000.0
# 需要投影的图像数量
BURNED_PATTERN_COUNT = 18
# USB 设备索引；只有一台投影仪时通常为 0
DEVICE_INDEX: int = 0
# 已烧录图案播放结束后是否关闭 LED。
LED_OFF_AFTER_FINISH: bool = True
# 已烧录图案的起始序号，对应 MA 命令的 image_start_index。
START_FRAME: int = 0
# 投影的灰度值
_PULSE_GRAY: int = 255
PROJECTOR_BRIGHTNESS: int = 100

from real_projector import (
    REPEAT_FRAMES,
    TRIGGER_START_FRAME,
    _TRIGGER_MODE,
    _burned_pattern_wait_s,
    close_projector,
    open_projector,
)

# ============================== 全局参数 ==============================
# 保持烧录图案的投影参数与 real_projector.py 一致。
CAPTURE_PATTERN_COUNT = BURNED_PATTERN_COUNT + 1

# 相机设备索引。如果左右物理相机的枚举顺序不同，请调整这些值。
RIGHT_CAMERA_INDEX = 0
LEFT_CAMERA_INDEX = 1

# 相机外部触发源。如果触发线连接到相机的其他输入线路，请更改为 LINE1/LINE2/LINE3。
CAMERA_TRIGGER_SOURCE = MV_TRIGGER_SOURCE_LINE0

# 在两台相机开始取流后等待一段时间，以确保在触发投影仪前采集状态稳定。
CAMERA_READY_DELAY_S = 0.30

# 配置完投影仪序列后，发送触发脉冲前的等待时间。
PROJECTOR_READY_DELAY_S = 0.30

# SDK 图像缓存节点数。使用足够的节点来缓冲短触发序列，同时允许 Python 从两台相机中拉取图像帧。
CAMERA_IMAGE_NODE_NUM = CAPTURE_PATTERN_COUNT + 4

# 联调测试捕获的图像保存在此处。
DATA_DIR = Path(__file__).resolve().parent / "data"
LEFT_PICTURE_DIR_NAME = "Left_picture"
RIGHT_PICTURE_DIR_NAME = "Right_picture"

# 每张捕获图像的相机帧等待超时时间（毫秒）。
CAPTURE_TIMEOUT_MS = 2000

# 投影仪触发与帧采集之间的延迟。
# 在硬件触发模式下请保持为 0，以便脚本能立即排空（拉取）触发的帧。
PROJECTOR_START_DELAY_S = 0.0

# 在投影仪触发前清空两台相机的 SDK 图像缓冲区。
# 这会移除相机等待投影仪时捕获的空白帧。
CLEAR_BUFFER_BEFORE_TRIGGER = True

# PROJECTOR_START_DELAY_S 之后额外丢弃的未保存帧数。
# 通常在清空 SDK 缓冲区时设为 0 就足够了，但这对时序微调很有用。
DISCARD_LEADING_FRAMES = 0

# 投影仪完成一次完整烧录图案序列的预估时间。
PROJECTION_WAIT_S = _burned_pattern_wait_s() * (
    CAPTURE_PATTERN_COUNT / BURNED_PATTERN_COUNT
)


# ============================== 相机辅助函数 ==============================
def open_hardware_trigger_camera(device_index: int) -> My_HKCamera:
    """打开一台相机，设置曝光，并等待外部触发帧。"""

    camera = My_HKCamera(device_index=device_index)
    camera.open()
    camera.set_exposure_time(EXPOSURE_TIME_US)
    camera.set_trigger_mode(True, source=CAMERA_TRIGGER_SOURCE)
    set_image_node_num(camera, CAMERA_IMAGE_NODE_NUM)
    camera.start_grabbing()
    return camera


def open_stereo_cameras() -> tuple[My_HKCamera, My_HKCamera]:
    """在触发投影仪之前打开左右相机。"""

    left_camera: Optional[My_HKCamera] = None
    try:
        left_camera = open_hardware_trigger_camera(LEFT_CAMERA_INDEX)
        right_camera = open_hardware_trigger_camera(RIGHT_CAMERA_INDEX)
        if CAMERA_READY_DELAY_S > 0:
            time.sleep(CAMERA_READY_DELAY_S)
        return left_camera, right_camera
    except Exception:
        if left_camera is not None:
            left_camera.close()
        raise


def set_image_node_num(camera: My_HKCamera, node_num: int) -> None:
    """在取流前设置 SDK 缓存节点数，以减少触发前空白帧的堆积。"""

    if camera.camera is None:
        raise CameraError("相机句柄未创建。请先调用 open()。")
    ret = camera.camera.MV_CC_SetImageNodeNum(int(node_num))
    if ret != MV_OK:
        print("警告：设置图像节点数失败，错误码=0x%08X" % ret)


def clear_image_buffer(camera: My_HKCamera) -> None:
    """清空 SDK 输出缓存，确保保存的帧从投影仪就绪后开始。"""

    if camera.camera is None:
        raise CameraError("相机句柄未创建。请先调用 open()。")
    ret = camera.camera.MV_CC_ClearImageBuffer()
    if ret != MV_OK:
        print("警告：清空图像缓冲区失败，错误码=0x%08X" % ret)


def clear_stereo_buffers(
    left_camera: My_HKCamera,
    right_camera: My_HKCamera,
) -> None:
    """清空两台相机的 SDK 输出缓存。"""

    clear_image_buffer(left_camera)
    clear_image_buffer(right_camera)


def discard_stereo_frames(
    left_camera: My_HKCamera,
    right_camera: My_HKCamera,
    count: int,
) -> None:
    """读取并丢弃两台相机的引导帧。"""

    for _ in range(int(count)):
        left_camera.get_frame(timeout_ms=CAPTURE_TIMEOUT_MS, output="mono")
        right_camera.get_frame(timeout_ms=CAPTURE_TIMEOUT_MS, output="mono")


def stop_camera_grabbing(camera: Optional[My_HKCamera]) -> None:
    """采集完成后立即停止取流。"""

    if camera is not None and camera.is_grabbing:
        camera.stop_grabbing()


def set_projector_brightness(projector: Projector) -> None:
    """Set projector brightness from PROJECTOR_BRIGHTNESS."""

    projector.set_light(PROJECTOR_BRIGHTNESS)


# ============================== 图像保存 ==============================
def create_next_capture_dirs(create_picture_dirs: bool = True) -> tuple[Path, Path, Path]:
    """Create data/<next_number>/Left_picture and Right_picture directories."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.name)
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_number = max(existing_numbers, default=0) + 1
    capture_dir = DATA_DIR / str(next_number)
    left_save_dir = capture_dir / LEFT_PICTURE_DIR_NAME
    right_save_dir = capture_dir / RIGHT_PICTURE_DIR_NAME
    capture_dir.mkdir(parents=True, exist_ok=False)
    if create_picture_dirs:
        left_save_dir.mkdir(parents=True, exist_ok=False)
        right_save_dir.mkdir(parents=True, exist_ok=False)
    return capture_dir, left_save_dir, right_save_dir


def build_pair_name_stem(frame_index: Optional[int] = None) -> str:
    """Create one simple time-random name stem for a left/right image pair."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    random_part = random.randint(1000, 9999)
    if frame_index is None:
        return "%s_%04d" % (timestamp, random_part)
    return "%03d_%s_%04d" % (int(frame_index), timestamp, random_part)


def build_image_path(save_dir: Path, pair_name_stem: str, side_suffix: str) -> Path:
    """Create a left or right image path from a shared pair name stem."""

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / ("%s_%s.bmp" % (pair_name_stem, side_suffix))


def save_image(image, save_dir: Path, pair_name_stem: str, side_suffix: str) -> Path:
    """Save one mono image with the shared pair name and side suffix."""

    image_path = build_image_path(save_dir, pair_name_stem, side_suffix)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise RuntimeError("保存图像失败: %s" % image_path)
    return image_path


def acquire_stereo_frames(
    left_camera: My_HKCamera,
    right_camera: My_HKCamera,
    count: int,
) -> tuple[list[tuple[object, object]], list[tuple[object, object]]]:
    """在保存到磁盘之前，从两台相机获取触发的单色帧。"""

    left_frames: list[tuple[object, object]] = []
    right_frames: list[tuple[object, object]] = []

    for image_index in range(int(count)):
        left_image, left_info = left_camera.get_frame(
            timeout_ms=CAPTURE_TIMEOUT_MS,
            output="mono",
        )
        right_image, right_info = right_camera.get_frame(
            timeout_ms=CAPTURE_TIMEOUT_MS,
            output="mono",
        )
        left_frames.append((left_image, left_info))
        right_frames.append((right_image, right_info))

    return left_frames, right_frames


def save_stereo_frames(
    left_frames: list[tuple[object, object]],
    right_frames: list[tuple[object, object]],
    left_save_dir: Path,
    right_save_dir: Path,
) -> None:
    """在相机停止采集后保存双目帧。"""

    for frame_index, ((left_image, _left_info), (right_image, _right_info)) in enumerate(
        zip(left_frames, right_frames)
    ):
        pair_name_stem = build_pair_name_stem(frame_index)
        save_image(left_image, left_save_dir, pair_name_stem, "l")
        save_image(right_image, right_save_dir, pair_name_stem, "r")

# ============================== 投影仪辅助函数 ==============================
def trigger_burned_pattern_sequence(projector: Projector) -> None:
    """使用 MA/TJ 流程触发一次完整的烧录图案序列。"""

    projector.set_trigger_mode(_TRIGGER_MODE, read_reply=False)
    projector.configure_trigger_sequence(
        repeat_frames=REPEAT_FRAMES,
        image_count=CAPTURE_PATTERN_COUNT,
        trigger_start_frame=TRIGGER_START_FRAME,
        image_start_index=START_FRAME,
        read_reply=False,
    )
    if PROJECTOR_READY_DELAY_S > 0:
        time.sleep(PROJECTOR_READY_DELAY_S)
    projector.send_hardware_trigger_pulse(_PULSE_GRAY)


# ============================== 联调测试 ==============================
def run_joint_test() -> None:
    """触发一次烧录条纹，并保存双目单色图像。"""

    left_camera: Optional[My_HKCamera] = None
    right_camera: Optional[My_HKCamera] = None
    projector: Optional[Projector] = None

    try:
        capture_dir, left_save_dir, right_save_dir = create_next_capture_dirs()
        print_device_list()
        left_camera, right_camera = open_stereo_cameras()
        projector = open_projector(DEVICE_INDEX)
        set_projector_brightness(projector)

        print("本次采集保存路径: %s" % capture_dir)
        print("左相机图像保存路径: %s" % left_save_dir)
        print("右相机图像保存路径: %s" % right_save_dir)

        if CLEAR_BUFFER_BEFORE_TRIGGER:
            clear_stereo_buffers(left_camera, right_camera)

        print("正在触发投影仪烧录图案序列 (单次)...")
        trigger_burned_pattern_sequence(projector)
        sequence_start = time.monotonic()

        if PROJECTOR_START_DELAY_S > 0:
            time.sleep(PROJECTOR_START_DELAY_S)
        if DISCARD_LEADING_FRAMES > 0:
            print("正在丢弃 %d 帧引导帧..." % DISCARD_LEADING_FRAMES)
            discard_stereo_frames(
                left_camera,
                right_camera,
                DISCARD_LEADING_FRAMES,
            )

        left_frames, right_frames = acquire_stereo_frames(
            left_camera,
            right_camera,
            CAPTURE_PATTERN_COUNT,
        )

        stop_camera_grabbing(left_camera)
        stop_camera_grabbing(right_camera)
        save_stereo_frames(left_frames, right_frames, left_save_dir, right_save_dir)

        remaining_wait_s = PROJECTION_WAIT_S - (time.monotonic() - sequence_start)
        if remaining_wait_s > 0:
            time.sleep(remaining_wait_s)

        print("联调测试结束。共捕获 %d 个双目图像对。" % CAPTURE_PATTERN_COUNT)

    except KeyboardInterrupt:
        print("接收到键盘中断信号。正在关闭设备...")
    except CameraError as exc:
        print("相机报错: %s" % exc)
        raise
    finally:
        stop_camera_grabbing(left_camera)
        stop_camera_grabbing(right_camera)
        if projector is not None:
            if LED_OFF_AFTER_FINISH:
                close_projector(projector)
            else:
                projector.close()
        if left_camera is not None:
            left_camera.close()
        if right_camera is not None:
            right_camera.close()
        print("设备已关闭。")


def main() -> None:
    run_joint_test()


if __name__ == "__main__":
    main()
