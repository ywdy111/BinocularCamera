from __future__ import annotations

import time
from typing import Optional

from My_Projector import (
    Projector,
    ProjectorColor,
    ProjectorMode,
    TJSTProjectorSDK,
    TriggerMode,
)

# DLL 路径；为 None 时自动从工程默认目录查找 TJSTProjectorApi.dll。
DLL_PATH: Optional[str] = None

# USB 设备索引；只有一台投影仪时通常为 0。
DEVICE_INDEX: int = 0

# 已烧录图案的起始序号，对应 MA 命令的 image_start_index。
START_FRAME: int = 0

# 本次要播放的已烧录图案数量。
BURNED_PATTERN_COUNT: int = 18

# 投影播放帧率，单位 Hz，用于估算等待播放完成的时间。
PROJECTOR_FPS: float = 6.0

# 每张图案重复播放的帧数，对应 MA 命令的 repeat_frames。
REPEAT_FRAMES: int = 2

# 触发起始帧，只能为 0 或 1。
TRIGGER_START_FRAME: int = 0

# 已烧录图案播放结束后是否关闭 LED。
LED_OFF_AFTER_FINISH: bool = True

# 固定参数：亮度始终为 100，触发灰度始终为 255，触发模式始终为普通模式。
_BRIGHTNESS: int = 100
_PULSE_GRAY: int = 255
_TRIGGER_MODE: TriggerMode = TriggerMode.NORMAL


def open_projector(index: Optional[int] = None) -> Projector:
    """通过 USB 设备索引打开投影仪，并按固定亮度保持常亮。"""

    target_index = DEVICE_INDEX if index is None else index
    sdk = TJSTProjectorSDK(DLL_PATH)
    devices = sdk.enum_devices()

    if not devices:
        raise RuntimeError("No projector device found. Check USB, power, and driver.")

    for device in devices:
        if device.index == target_index:
            projector = sdk.open_index(device.index)
            _prepare_projector(projector)
            return projector

    available = ", ".join(str(device.index) for device in devices)
    raise ValueError(
        f"Projector index {target_index} was not found. Available: {available}"
    )


def close_projector(projector: Optional[Projector]) -> None:
    """关闭 LED 并释放投影仪连接。"""

    if projector is None:
        return

    try:
        projector.led_off()
    finally:
        projector.close()


def _prepare_projector(projector: Projector) -> None:
    """设置亮度、颜色、LED 和基础白场模式。"""

    projector.set_light(_BRIGHTNESS)
    projector.set_color(ProjectorColor.WHITE)
    projector.led_on()
    projector.set_mode(ProjectorMode.WHITE)
    time.sleep(0.2)


def _burned_pattern_wait_s() -> float:
    """根据图案数量、重复帧数和帧率估算播放等待时间。"""

    if BURNED_PATTERN_COUNT <= 0:
        raise ValueError("BURNED_PATTERN_COUNT must be greater than 0.")
    if PROJECTOR_FPS <= 0:
        raise ValueError("PROJECTOR_FPS must be greater than 0.")

    return (REPEAT_FRAMES + 1) * PROJECTOR_FPS/BURNED_PATTERN_COUNT


def project_burned_patterns(projector: Optional[Projector] = None) -> Optional[Projector]:
    """播放一轮投影仪内部已烧录图案。

    不传入 projector 时，本函数会自动打开并关闭 USB 投影仪。
    传入已打开的 projector 时，本函数只负责播放，连接由调用者关闭。
    """

    own_projector = projector is None
    projector = open_projector() if projector is None else projector

    try:
        if not own_projector:
            _prepare_projector(projector)
        projector.set_trigger_mode(_TRIGGER_MODE, read_reply=False)
        projector.configure_trigger_sequence(
            repeat_frames=REPEAT_FRAMES,
            image_count=BURNED_PATTERN_COUNT,
            trigger_start_frame=TRIGGER_START_FRAME,
            image_start_index=START_FRAME,
            read_reply=False,
        )
        projector.send_hardware_trigger_pulse(_PULSE_GRAY)
        time.sleep(_burned_pattern_wait_s())

        if LED_OFF_AFTER_FINISH and not own_projector:
            projector.led_off()

        return None if own_projector else projector
    finally:
        if own_projector:
            if LED_OFF_AFTER_FINISH:
                try:
                    projector.led_off()
                except Exception:
                    pass
            projector.close()


def project_chessboard(projector: Optional[Projector] = None) -> Projector:
    """投影内置棋盘格图案，投影完成后保持投影仪打开。"""

    own_projector = projector is None
    projector = open_projector() if projector is None else projector

    try:
        if not own_projector:
            _prepare_projector(projector)
        projector.project_chessboard()
        return projector
    except Exception:
        if own_projector:
            close_projector(projector)
        raise


def project_cross(projector: Optional[Projector] = None) -> Projector:
    """投影内置十字线图案，投影完成后保持投影仪打开。"""

    own_projector = projector is None
    projector = open_projector() if projector is None else projector

    try:
        if not own_projector:
            _prepare_projector(projector)
        projector.project_cross()
        return projector
    except Exception:
        if own_projector:
            close_projector(projector)
        raise


def _show_message(title: str, message: str) -> None:
    """使用系统对话框显示运行结果。"""

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


def _show_error(title: str, message: str) -> None:
    """使用系统错误对话框显示异常信息。"""

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def demo_all_functions_with_dialogs() -> int:
    """依次调用本文件的主要函数，并用对话框显示每一步结果。"""

    projector: Optional[Projector] = None

    try:
        projector = open_projector()
        _show_message("投影仪", "open_projector() 调用成功，投影仪已打开。")

        project_chessboard(projector)
        _show_message("投影仪", "project_chessboard() 调用成功，已投影棋盘格。")

        project_cross(projector)
        _show_message("投影仪", "project_cross() 调用成功，已投影十字线。")

        project_burned_patterns(projector)
        
        _show_message(
            "投影仪",
            "project_burned_patterns() 调用成功，已播放一轮已烧录图案。",
        )

        close_projector(projector)
        projector = None
        _show_message("投影仪", "close_projector() 调用成功，投影仪已关闭。")

        return 0
    except KeyboardInterrupt:
        close_projector(projector)
        return 130
    except Exception as exc:
        close_projector(projector)
        _show_error("投影仪错误", str(exc))
        return 1


def run() -> int:
    """脚本入口：通过对话框演示各个函数调用。"""

    return demo_all_functions_with_dialogs()


if __name__ == "__main__":
    raise SystemExit(run())
