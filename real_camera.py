#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""海康相机交互测试脚本。

功能：
1. 使用全局变量 EXPOSURE_TIME_US 设置曝光时间。
2. 实时显示相机画面。
3. 按 s 保存当前画面。
4. 按 q 停止取流、断开相机并退出程序。

需要的外部库：
- opencv-python
- numpy
"""

from datetime import datetime
from pathlib import Path

import cv2

from My_HKCamera import CameraError, My_HKCamera


# ============================== 全局参数配置 ==============================
# 曝光时间，单位：微秒。根据现场亮度修改这个值即可。
EXPOSURE_TIME_US = 10000.0

# 默认打开第 0 台相机。如果有多台相机，可改成 1、2...
CAMERA_INDEX = 0

# 拍照保存目录。
SAVE_DIR = Path(__file__).resolve().parent / "captures"

# OpenCV 显示窗口名称。
WINDOW_NAME = "HK Camera Test"

# 显示比例仅供预览, 保存的图像将保留原始分辨率。
DISPLAY_SCALE = 0.8


# ============================== 相机初始化 ==============================
def print_device_list() -> None:
    """打印当前在线相机列表，便于确认 CAMERA_INDEX 对应哪台设备。"""

    devices = My_HKCamera.enumerate_devices()
    print("Found %d camera(s)" % len(devices))
    for device in devices:
        print(
            "[{index}] model={model} serial={serial} name={name} ip={ip}".format(
                index=device.index,
                model=device.model_name,
                serial=device.serial_number,
                name=device.user_defined_name,
                ip=device.ip,
            )
        )


def open_camera() -> My_HKCamera:
    """打开相机并设置曝光。"""

    camera = My_HKCamera(device_index=CAMERA_INDEX)
    camera.open()
    camera.set_exposure_time(EXPOSURE_TIME_US)
    camera.start_grabbing()
    return camera


# ============================== 图像保存 ==============================
def build_image_path(frame_info) -> Path:
    """生成带时间、曝光和帧号的图片文件名。"""

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    frame_num = getattr(frame_info, "nFrameNum", 0)
    filename = "capture_%s_exp_%sus_frame_%s.png" % (
        timestamp,
        int(EXPOSURE_TIME_US),
        frame_num,
    )
    return SAVE_DIR / filename


def save_image(image, frame_info) -> None:
    """保存当前帧。"""

    image_path = build_image_path(frame_info)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise RuntimeError("Save image failed: %s" % image_path)
    print("Saved: %s" % image_path)


# ============================== 显示窗口 ==============================
def resize_for_display(image):
    """Return a scaled preview image without changing the original capture."""

    if DISPLAY_SCALE == 1.0:
        return image
    return cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA,
    )


# ============================== 主循环与按键控制 ==============================
def main() -> None:
    camera = None

    try:
        print_device_list()
        camera = open_camera()

        print("Camera opened. ExposureTime = %s us" % EXPOSURE_TIME_US)
        print("Press s to save image, press q to close camera and quit.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            image, frame_info = camera.get_frame(timeout_ms=1000, output="bgr")
            preview = resize_for_display(image)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_image(image, frame_info)
            elif key == ord("q"):
                print("Quit requested. Closing camera...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt. Closing camera...")
    except CameraError as exc:
        print("Camera error: %s" % exc)
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
        print("Camera disconnected. Program exited.")


if __name__ == "__main__":
    main()
