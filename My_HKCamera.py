#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""High level helper for Hikvision MVS cameras.

This module wraps the low-level ctypes SDK files in this project and exposes a
small Pythonic camera class. It supports device enumeration, opening a camera,
starting/stopping acquisition, reading frames as NumPy arrays, and basic camera
parameter control.
"""

from __future__ import annotations

import os
import platform
import sys
from ctypes import POINTER, addressof, c_ubyte, memset, sizeof
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================== 模块一：SDK 动态库路径配置 ==============================
# 在导入海康 MVS ctypes 封装前，先把项目自带的 DLL 目录加入系统搜索路径。
_PROJECT_DIR = Path(__file__).resolve().parent
_DLL_DIR_HANDLE = None


def _configure_sdk_dll_path() -> Path:
    """Add the bundled MVS DLL directory to the process search path."""

    if platform.system() != "Windows":
        return _PROJECT_DIR / "lib/camera"

    arch_dir = "Win64_x64" if sys.maxsize > 2**32 else "Win32_i86"
    dll_dir = _PROJECT_DIR / "lib/camera" / arch_dir
    if not dll_dir.exists():
        raise FileNotFoundError("MVS DLL directory not found: %s" % dll_dir)

    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

    global _DLL_DIR_HANDLE
    if hasattr(os, "add_dll_directory") and _DLL_DIR_HANDLE is None:
        _DLL_DIR_HANDLE = os.add_dll_directory(str(dll_dir))

    return dll_dir


_SDK_DLL_DIR = _configure_sdk_dll_path()

# ============================== 模块二：SDK 常量、结构体、底层类导入 ==============================
# 这些文件来自海康 MVS Python SDK 示例，负责提供设备枚举、取流、像素转换等底层接口。
from config.camera.CameraParams_const import (  # noqa: E402
    MV_ACCESS_Exclusive,
    MV_GIGE_DEVICE,
    MV_GENTL_GIGE_DEVICE,
    MV_USB_DEVICE,
)
from config.camera.CameraParams_header import (  # noqa: E402
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE_EX,
    MVCC_STRINGVALUE,
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM,
    MV_FRAME_OUT,
    MV_TRIGGER_MODE_OFF,
    MV_TRIGGER_MODE_ON,
    MV_TRIGGER_SOURCE_SOFTWARE,
)
from config.camera.MvCameraControl_class import MvCamera  # noqa: E402
from config.camera.MvErrorDefine_const import MV_E_NODATA, MV_OK  # noqa: E402
from config.camera.PixelType_header import (  # noqa: E402
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_BGRA8_Packed,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono16,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_RGBA8_Packed,
)


# ============================== 模块三：第三方依赖检查 ==============================
# NumPy 用于把 SDK 返回的图像缓存转换成 ndarray；缺失时在取图阶段抛出明确错误。
try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    np = None


# ============================== 模块四：异常与设备信息数据结构 ==============================
class CameraError(RuntimeError):
    """Raised when the MVS SDK returns an error code."""


@dataclass(frozen=True)
class CameraDevice:
    index: int
    transport_layer: int
    model_name: str
    serial_number: str
    user_defined_name: str
    ip: str = ""


# ============================== 模块五：内部辅助函数 ==============================
# 这里集中处理 SDK 错误码、C 字符数组解析、IP 转换、帧宽高获取等边界细节。
def _hex(code: int) -> str:
    return "0x%08X" % (code & 0xFFFFFFFF)


def _check(code: int, action: str, allow_no_data: bool = False) -> None:
    if code == MV_OK:
        return
    if allow_no_data and code == MV_E_NODATA:
        return
    raise CameraError("%s failed, error=%s" % (action, _hex(code)))


def _c_ubyte_array_to_str(value) -> str:
    data = bytes(value)
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def _ip_to_str(ip_value: int) -> str:
    if not ip_value:
        return ""
    return ".".join(str((ip_value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _frame_width(info) -> int:
    return int(info.nExtendWidth or info.nWidth)


def _frame_height(info) -> int:
    return int(info.nExtendHeight or info.nHeight)


def _require_numpy():
    if np is None:
        raise ImportError("NumPy is required to return images. Install numpy first.")
    return np


# ============================== 模块六：相机高层封装类 ==============================
# 对外主要使用 My_HKCamera，实现枚举、打开、取流、采图、参数读写和释放资源。
class My_HKCamera:
    """Convenience wrapper around Hikvision MVS ``MvCamera``.

    Example:
        camera = My_HKCamera.open_first()
        image, info = camera.get_frame()
        camera.close()
    """

    DEFAULT_TLAYER_TYPE = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_GIGE_DEVICE

    def __init__(self, device_index: int = 0, access_mode: int = MV_ACCESS_Exclusive):
        self.device_index = device_index
        self.access_mode = access_mode
        self.camera: Optional[MvCamera] = None
        self.device_info: Optional[MV_CC_DEVICE_INFO] = None
        self.is_open = False
        self.is_grabbing = False

    def __enter__(self) -> "My_HKCamera":
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def sdk_dll_dir() -> Path:
        return _SDK_DLL_DIR

    # ------------------------------ 6.1 设备枚举与打开 ------------------------------
    @classmethod
    def enumerate_devices(cls, tlayer_type: int = DEFAULT_TLAYER_TYPE) -> List[CameraDevice]:
        st_device_list = MV_CC_DEVICE_INFO_LIST()
        memset(addressof(st_device_list), 0, sizeof(st_device_list))

        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, st_device_list)
        _check(ret, "Enum devices")

        devices: List[CameraDevice] = []
        for index in range(int(st_device_list.nDeviceNum)):
            raw_info = st_device_list.pDeviceInfo[index].contents
            info = raw_info.SpecialInfo

            if raw_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                gige_info = info.stGigEInfo
                device = CameraDevice(
                    index=index,
                    transport_layer=int(raw_info.nTLayerType),
                    model_name=_c_ubyte_array_to_str(gige_info.chModelName),
                    serial_number=_c_ubyte_array_to_str(gige_info.chSerialNumber),
                    user_defined_name=_c_ubyte_array_to_str(gige_info.chUserDefinedName),
                    ip=_ip_to_str(int(gige_info.nCurrentIp)),
                )
            elif raw_info.nTLayerType == MV_USB_DEVICE:
                usb_info = info.stUsb3VInfo
                device = CameraDevice(
                    index=index,
                    transport_layer=int(raw_info.nTLayerType),
                    model_name=_c_ubyte_array_to_str(usb_info.chModelName),
                    serial_number=_c_ubyte_array_to_str(usb_info.chSerialNumber),
                    user_defined_name=_c_ubyte_array_to_str(usb_info.chUserDefinedName),
                )
            else:
                device = CameraDevice(
                    index=index,
                    transport_layer=int(raw_info.nTLayerType),
                    model_name="Unknown",
                    serial_number="",
                    user_defined_name="",
                )

            devices.append(device)

        return devices

    @classmethod
    def open_first(cls, **kwargs) -> "My_HKCamera":
        camera = cls(device_index=0, **kwargs)
        camera.open()
        return camera

    @classmethod
    def open_by_serial(cls, serial_number: str, **kwargs) -> "My_HKCamera":
        devices = cls.enumerate_devices()
        for device in devices:
            if device.serial_number == serial_number:
                camera = cls(device_index=device.index, **kwargs)
                camera.open()
                return camera
        raise CameraError("Camera serial number not found: %s" % serial_number)

    def open(self) -> None:
        if self.is_open:
            return

        st_device_list = MV_CC_DEVICE_INFO_LIST()
        memset(addressof(st_device_list), 0, sizeof(st_device_list))
        ret = MvCamera.MV_CC_EnumDevices(self.DEFAULT_TLAYER_TYPE, st_device_list)
        _check(ret, "Enum devices")

        if st_device_list.nDeviceNum == 0:
            raise CameraError("No Hikvision MVS camera found.")
        if self.device_index < 0 or self.device_index >= st_device_list.nDeviceNum:
            raise IndexError(
                "Camera index %s out of range, found %s device(s)."
                % (self.device_index, st_device_list.nDeviceNum)
            )

        self.device_info = st_device_list.pDeviceInfo[self.device_index].contents
        self.camera = MvCamera()

        ret = self.camera.MV_CC_CreateHandle(self.device_info)
        _check(ret, "Create handle")

        try:
            ret = self.camera.MV_CC_OpenDevice(self.access_mode, 0)
            _check(ret, "Open device")

            if self.device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                packet_size = self.camera.MV_CC_GetOptimalPacketSize()
                if packet_size > 0:
                    _check(
                        self.camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size),
                        "Set GevSCPSPacketSize",
                    )

            self.set_trigger_mode(False)
            self.is_open = True
        except Exception:
            self._destroy_handle()
            raise

    # ------------------------------ 6.2 关闭与资源释放 ------------------------------
    def close(self) -> None:
        if self.camera is None:
            return

        if self.is_grabbing:
            self.stop_grabbing()

        if self.is_open:
            _check(self.camera.MV_CC_CloseDevice(), "Close device")
            self.is_open = False

        self._destroy_handle()

    def _destroy_handle(self) -> None:
        if self.camera is not None:
            self.camera.MV_CC_DestroyHandle()
        self.camera = None
        self.device_info = None
        self.is_open = False
        self.is_grabbing = False

    # ------------------------------ 6.3 取流与图像采集 ------------------------------
    def start_grabbing(self) -> None:
        self._ensure_open()
        if self.is_grabbing:
            return
        _check(self.camera.MV_CC_StartGrabbing(), "Start grabbing")
        self.is_grabbing = True

    def stop_grabbing(self) -> None:
        self._ensure_camera()
        if not self.is_grabbing:
            return
        _check(self.camera.MV_CC_StopGrabbing(), "Stop grabbing")
        self.is_grabbing = False

    def get_frame(self, timeout_ms: int = 1000, output: str = "bgr") -> Tuple["np.ndarray", object]:
        """Grab one frame.

        Args:
            timeout_ms: SDK wait timeout in milliseconds.
            output: ``"bgr"`` (default), ``"rgb"``, ``"mono"``, or ``"raw"``.

        Returns:
            ``(image, frame_info)``. ``image`` is copied before the SDK buffer is
            released, so it remains valid after this method returns.
        """

        npx = _require_numpy()
        self._ensure_open()
        if not self.is_grabbing:
            self.start_grabbing()

        frame = MV_FRAME_OUT()
        memset(addressof(frame), 0, sizeof(frame))

        ret = self.camera.MV_CC_GetImageBuffer(frame, timeout_ms)
        _check(ret, "Get image buffer")

        try:
            info = frame.stFrameInfo
            width = _frame_width(info)
            height = _frame_height(info)
            pixel_type = int(info.enPixelType)
            data_len = int(info.nFrameLenEx or info.nFrameLen)
            raw = npx.ctypeslib.as_array(frame.pBufAddr, shape=(data_len,)).copy()

            output = output.lower()
            if output == "raw":
                return raw, info
            if output == "mono":
                return self._as_mono(raw, width, height, pixel_type), info
            if output not in ("bgr", "rgb"):
                raise ValueError("output must be one of: bgr, rgb, mono, raw")

            image = self._as_bgr(raw, width, height, pixel_type)
            if output == "rgb":
                image = image[:, :, ::-1].copy()
            return image, info
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame)

    # ------------------------------ 6.4 常用参数设置 ------------------------------
    def software_trigger(self) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetCommandValue("TriggerSoftware"), "Software trigger")

    def set_trigger_mode(self, enabled: bool, source: int = MV_TRIGGER_SOURCE_SOFTWARE) -> None:
        self._ensure_camera()
        mode = MV_TRIGGER_MODE_ON if enabled else MV_TRIGGER_MODE_OFF
        _check(self.camera.MV_CC_SetEnumValue("TriggerMode", mode), "Set TriggerMode")
        if enabled:
            _check(self.camera.MV_CC_SetEnumValue("TriggerSource", source), "Set TriggerSource")

    def set_exposure_time(self, exposure_us: float) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetEnumValue("ExposureAuto", 0), "Disable ExposureAuto")
        _check(self.camera.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)), "Set ExposureTime")

    def set_gain(self, gain: float) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetEnumValue("GainAuto", 0), "Disable GainAuto")
        _check(self.camera.MV_CC_SetFloatValue("Gain", float(gain)), "Set Gain")

    def set_frame_rate(self, fps: float) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), "Enable frame rate")
        _check(self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps)), "Set frame rate")

    def set_enum(self, key: str, value: int) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetEnumValue(key, int(value)), "Set %s" % key)

    def set_float(self, key: str, value: float) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetFloatValue(key, float(value)), "Set %s" % key)

    def set_int(self, key: str, value: int) -> None:
        self._ensure_open()
        _check(self.camera.MV_CC_SetIntValueEx(key, int(value)), "Set %s" % key)

    # ------------------------------ 6.5 常用参数读取 ------------------------------
    def get_int(self, key: str) -> Dict[str, int]:
        self._ensure_open()
        value = MVCC_INTVALUE_EX()
        memset(addressof(value), 0, sizeof(value))
        _check(self.camera.MV_CC_GetIntValueEx(key, value), "Get %s" % key)
        return {
            "current": int(value.nCurValue),
            "max": int(value.nMax),
            "min": int(value.nMin),
            "inc": int(value.nInc),
        }

    def get_float(self, key: str) -> Dict[str, float]:
        self._ensure_open()
        value = MVCC_FLOATVALUE()
        memset(addressof(value), 0, sizeof(value))
        _check(self.camera.MV_CC_GetFloatValue(key, value), "Get %s" % key)
        return {
            "current": float(value.fCurValue),
            "max": float(value.fMax),
            "min": float(value.fMin),
        }

    def get_enum(self, key: str) -> Dict[str, object]:
        self._ensure_open()
        value = MVCC_ENUMVALUE()
        memset(addressof(value), 0, sizeof(value))
        _check(self.camera.MV_CC_GetEnumValue(key, value), "Get %s" % key)
        supported = [int(value.nSupportValue[i]) for i in range(int(value.nSupportedNum))]
        return {"current": int(value.nCurValue), "supported": supported}

    def get_string(self, key: str) -> str:
        self._ensure_open()
        value = MVCC_STRINGVALUE()
        memset(addressof(value), 0, sizeof(value))
        _check(self.camera.MV_CC_GetStringValue(key, value), "Get %s" % key)
        return bytes(value.chCurValue).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    # ------------------------------ 6.6 图像格式转换 ------------------------------
    def _as_mono(self, raw, width: int, height: int, pixel_type: int):
        npx = _require_numpy()
        if pixel_type == PixelType_Gvsp_Mono8:
            return raw[: width * height].reshape(height, width)
        if pixel_type == PixelType_Gvsp_Mono16:
            return raw.view(npx.uint16)[: width * height].reshape(height, width)
        return self._convert(raw, width, height, pixel_type, PixelType_Gvsp_Mono8).reshape(height, width)

    def _as_bgr(self, raw, width: int, height: int, pixel_type: int):
        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            return raw[: width * height * 3].reshape(height, width, 3)
        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            return raw[: width * height * 3].reshape(height, width, 3)[:, :, ::-1].copy()
        if pixel_type == PixelType_Gvsp_BGRA8_Packed:
            return raw[: width * height * 4].reshape(height, width, 4)[:, :, :3].copy()
        if pixel_type == PixelType_Gvsp_RGBA8_Packed:
            return raw[: width * height * 4].reshape(height, width, 4)[:, :, [2, 1, 0]].copy()
        if pixel_type == PixelType_Gvsp_Mono8:
            mono = raw[: width * height].reshape(height, width)
            return self._gray_to_bgr(mono)

        bgr = self._convert(raw, width, height, pixel_type, PixelType_Gvsp_BGR8_Packed)
        return bgr.reshape(height, width, 3)

    def _convert(self, raw, width: int, height: int, src_pixel_type: int, dst_pixel_type: int):
        npx = _require_numpy()
        dst_size = width * height * (3 if dst_pixel_type == PixelType_Gvsp_BGR8_Packed else 1)
        dst = npx.empty(dst_size, dtype=npx.uint8)

        convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(addressof(convert_param), 0, sizeof(convert_param))
        convert_param.nWidth = width
        convert_param.nHeight = height
        convert_param.enSrcPixelType = src_pixel_type
        convert_param.pSrcData = raw.ctypes.data_as(POINTER(c_ubyte))
        convert_param.nSrcDataLen = int(raw.nbytes)
        convert_param.enDstPixelType = dst_pixel_type
        convert_param.pDstBuffer = dst.ctypes.data_as(POINTER(c_ubyte))
        convert_param.nDstBufferSize = int(dst.nbytes)

        _check(self.camera.MV_CC_ConvertPixelType(convert_param), "Convert pixel type")
        return dst[: int(convert_param.nDstLen or dst_size)].copy()

    @staticmethod
    def _gray_to_bgr(mono):
        npx = _require_numpy()
        return npx.stack((mono, mono, mono), axis=2)

    # ------------------------------ 6.7 状态检查 ------------------------------
    def _ensure_camera(self) -> None:
        if self.camera is None:
            raise CameraError("Camera handle has not been created. Call open() first.")

    def _ensure_open(self) -> None:
        self._ensure_camera()
        if not self.is_open:
            raise CameraError("Camera is not open. Call open() first.")


# ============================== 模块七：兼容别名与命令行入口 ==============================
HKCamera = My_HKCamera


def list_devices() -> List[CameraDevice]:
    return My_HKCamera.enumerate_devices()


def _print_devices() -> None:
    devices = list_devices()
    print("SDK DLL directory: %s" % My_HKCamera.sdk_dll_dir())
    print("Found %d camera(s)" % len(devices))
    for device in devices:
        print(
            "[{index}] model={model} serial={serial} name={name} ip={ip} tlayer={tlayer}".format(
                index=device.index,
                model=device.model_name,
                serial=device.serial_number,
                name=device.user_defined_name,
                ip=device.ip,
                tlayer=device.transport_layer,
            )
        )


if __name__ == "__main__":
    _print_devices()
