"""Python wrapper for TJSTProjectorApi.dll.

The C API is declared in:
    TJSTProjectorApiSDK/Include/TJSTProjectorApi.h

This module keeps the ctypes layer small and explicit, while providing a
friendlier Projector class for normal use.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "BinocularCamera" / "lib" / "TJSTP"
DEFAULT_DLL_DIRS = (
    SDK_ROOT / "x64" / "Release",
    SDK_ROOT / "TJSTProjectorApiSDK" / "Release",
    SDK_ROOT / "x64" / "Debug",
    SDK_ROOT / "TJSTProjectorApiSDK" / "Debug",
)
DEFAULT_DLL_NAME = "TJSTProjectorApi.dll"
DEFAULT_NET_PORT = 1234
DEFAULT_PATTERN_WIDTH = 1280
MAX_CUSTOM_PATTERN_FRAMES = 100
RUNTIME_HINT = (
    "TJSTProjectorApi.dll depends on EasyPODx64.dll, VC++ 2015 runtime "
    "(MSVCP140.dll/VCRUNTIME140.dll) and VC++ 2010 MFC runtime "
    "(mfc100.dll/MSVCR100.dll). The SDK includes "
    "TJSTProjectorApiSDK/vs2010运行库/vcredist_x64.exe for the 2010 runtime."
)


class ProjectorError(RuntimeError):
    """Raised when the projector SDK reports an operation failure."""


class ProjectorType(IntEnum):
    USBPOD = 0
    USBHID = 1
    NET = 2


class ProjectorMode(IntEnum):
    BLACK = 0
    WHITE = 1
    CROSS = 2
    CHESSBOARD = 3


class BuiltInImage(IntEnum):
    BLACK = 0
    WHITE = 1
    CROSS = 2
    CHESSBOARD = 3
    CUSTOM_1 = 4
    CROSS_2 = 6
    CROSS_3 = 7


class ProjectorColor(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    WHITE = 3


class ImageRotation(IntEnum):
    NONE = 0
    X = 1
    Y = 2
    XY = 3


class TriggerMode(IntEnum):
    NORMAL = 0
    LOOP = 1
    SINGLE_FRAME_HOLD = 2
    OBLIQUE = 4


class LightSourceMode(IntEnum):
    STANDARD = 0
    EXTERNAL_1 = 1
    EXTERNAL_2 = 2
    TRIGGER_120HZ = 3
    TRIGGER_240HZ = 4
    TRIGGER_360HZ = 5


class ObliqueDirection(IntEnum):
    DEFAULT = 0
    ALTERNATE = 1


class _TJSTPrjInfoUnion(ctypes.Union):
    _fields_ = [
        ("prjID", ctypes.c_int),
        ("prjIP", ctypes.c_ubyte * 4),
    ]


class TJSTPrjInfo(ctypes.Structure):
    _fields_ = [
        ("prjType", ctypes.c_int),
        ("prjIndex", ctypes.c_int),
        ("prjInfo", _TJSTPrjInfoUnion),
        ("prjVer", ctypes.c_char * 64),
    ]


@dataclass(frozen=True)
class ProjectorInfo:
    index: int
    type: ProjectorType
    device_index: int
    device_id: Optional[int]
    ip: Optional[str]
    version: str

    @classmethod
    def from_c(cls, index: int, raw: TJSTPrjInfo) -> "ProjectorInfo":
        prj_type = ProjectorType(raw.prjType)
        ip = None
        device_id = None
        if prj_type == ProjectorType.NET:
            ip = ".".join(str(raw.prjInfo.prjIP[i]) for i in range(4))
        else:
            device_id = int(raw.prjInfo.prjID)

        version = bytes(raw.prjVer).split(b"\0", 1)[0].decode(
            "gbk", errors="replace"
        )
        return cls(
            index=index,
            type=prj_type,
            device_index=int(raw.prjIndex),
            device_id=device_id,
            ip=ip,
            version=version,
        )

    def __str__(self) -> str:
        if self.type == ProjectorType.NET:
            address = f"ip={self.ip}"
        else:
            address = f"id={self.device_id}"
        return (
            f"[{self.index}] {self.type.name} device_index={self.device_index} "
            f"{address} version={self.version}"
        )


def _candidate_dlls(dll_path: Optional[os.PathLike[str] | str]) -> Iterable[Path]:
    if dll_path:
        yield Path(dll_path)
        return
    for dll_dir in DEFAULT_DLL_DIRS:
        yield dll_dir / DEFAULT_DLL_NAME


def find_dll(dll_path: Optional[os.PathLike[str] | str] = None) -> Path:
    for path in _candidate_dlls(dll_path):
        if path.is_file():
            return path.resolve()
    searched = "\n".join(str(path) for path in _candidate_dlls(dll_path))
    raise FileNotFoundError(f"Cannot find {DEFAULT_DLL_NAME}. Searched:\n{searched}")


def _load_library(dll_path: Optional[os.PathLike[str] | str] = None) -> ctypes.CDLL:
    path = find_dll(dll_path)
    dependency_dirs = [path.parent]
    if path.parent.name.lower() in {"release", "debug"}:
        dependency_dirs.append(path.parent.parent)
    dependency_dirs.extend([SDK_ROOT / "Dll", PROJECT_ROOT / "x64" / "Release"])

    if hasattr(os, "add_dll_directory"):
        for dependency_dir in dependency_dirs:
            if dependency_dir.is_dir():
                os.add_dll_directory(str(dependency_dir))
    path_entries = [str(p) for p in dependency_dirs if p.is_dir()]
    os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        search_text = "\n".join(path_entries)
        raise OSError(
            f"{exc}\nDLL search paths:\n{search_text}\nDependency hint: {RUNTIME_HINT}"
        ) from exc


class TJSTProjectorSDK:
    """ctypes binding for the exported TJST projector SDK functions."""

    def __init__(self, dll_path: Optional[os.PathLike[str] | str] = None):
        self.dll_path = find_dll(dll_path)
        self.dll = _load_library(self.dll_path)
        self._bind_functions()

    def _bind_functions(self) -> None:
        dll = self.dll

        dll.TJSTEnumDevices.argtypes = []
        dll.TJSTEnumDevices.restype = ctypes.c_int

        dll.TJSTGetPrjInfo.argtypes = [ctypes.c_int]
        dll.TJSTGetPrjInfo.restype = ctypes.POINTER(TJSTPrjInfo)

        self._bind_optional("TJSTPrjOpenIndex", [ctypes.c_int], ctypes.c_void_p)

        dll.TJSTPrjOpen.argtypes = [ctypes.POINTER(TJSTPrjInfo)]
        dll.TJSTPrjOpen.restype = ctypes.c_void_p

        dll.TJSTNetPrjOpen.argtypes = [ctypes.c_char_p]
        dll.TJSTNetPrjOpen.restype = ctypes.c_void_p

        self._bind_optional(
            "TJSTNetPrjOpenEx", [ctypes.c_char_p, ctypes.c_int], ctypes.c_void_p
        )

        dll.TJSTPrjClose.argtypes = [ctypes.c_void_p]
        dll.TJSTPrjClose.restype = None

        dll.TJSTPrjWrite.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        dll.TJSTPrjWrite.restype = ctypes.c_int

        self._bind_optional("TJSTPrjCmdNotBack", [ctypes.c_void_p], None)

        dll.TJSTPrjRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        dll.TJSTPrjRead.restype = ctypes.c_int

        dll.TJSTPrjLedOn.argtypes = [ctypes.c_void_p]
        dll.TJSTPrjLedOn.restype = ctypes.c_bool

        dll.TJSTPrjLedOff.argtypes = [ctypes.c_void_p]
        dll.TJSTPrjLedOff.restype = ctypes.c_bool

        dll.TJSTPrjSetMode.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        dll.TJSTPrjSetMode.restype = ctypes.c_bool

        dll.TJSTPrjSetColor.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        dll.TJSTPrjSetColor.restype = ctypes.c_bool

        dll.TJSTPrjSetLight.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        dll.TJSTPrjSetLight.restype = ctypes.c_bool

        dll.TJSTPrjTriggerOnce.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        dll.TJSTPrjTriggerOnce.restype = ctypes.c_bool

    def _bind_optional(self, name: str, argtypes: list, restype) -> bool:
        try:
            func = getattr(self.dll, name)
        except AttributeError:
            return False
        func.argtypes = argtypes
        func.restype = restype
        return True

    def enum_devices(self) -> list[ProjectorInfo]:
        count = self.dll.TJSTEnumDevices()
        if count < 0:
            raise ProjectorError(f"TJSTEnumDevices returned {count}")

        devices: list[ProjectorInfo] = []
        for index in range(count):
            info_ptr = self.dll.TJSTGetPrjInfo(index)
            if not info_ptr:
                continue
            devices.append(ProjectorInfo.from_c(index, info_ptr.contents))
        return devices

    def open_index(self, index: int) -> "Projector":
        if not hasattr(self.dll, "TJSTPrjOpenIndex"):
            return self.open_info(index)
        handle = self.dll.TJSTPrjOpenIndex(int(index))
        if not handle:
            raise ProjectorError(f"Cannot open projector index {index}")
        return Projector(self, handle)

    def open_info(self, index: int) -> "Projector":
        info_ptr = self.dll.TJSTGetPrjInfo(int(index))
        if not info_ptr:
            raise ProjectorError(f"Cannot get projector info for index {index}")
        handle = self.dll.TJSTPrjOpen(info_ptr)
        if not handle:
            raise ProjectorError(f"Cannot open projector index {index}")
        return Projector(self, handle)

    def open_net(self, ip: str, port: int = DEFAULT_NET_PORT) -> "Projector":
        ip_bytes = ip.encode("ascii")
        if int(port) == DEFAULT_NET_PORT or not hasattr(self.dll, "TJSTNetPrjOpenEx"):
            handle = self.dll.TJSTNetPrjOpen(ip_bytes)
        else:
            handle = self.dll.TJSTNetPrjOpenEx(ip_bytes, int(port))
        if not handle:
            raise ProjectorError(f"Cannot open network projector {ip}:{port}")
        return Projector(self, handle)


class Projector:
    """High-level projector handle.

    Use this class as a context manager so the DLL handle is always closed:

        sdk = TJSTProjectorSDK()
        with sdk.open_index(0) as prj:
            prj.led_on()
            prj.set_light(50)
    """

    def __init__(self, sdk: TJSTProjectorSDK, handle: int):
        self.sdk = sdk
        self.handle = ctypes.c_void_p(handle)
        self._closed = False

    def __enter__(self) -> "Projector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed and self.handle:
            self.sdk.dll.TJSTPrjClose(self.handle)
            self._closed = True
            self.handle = ctypes.c_void_p()

    def _require_open(self) -> ctypes.c_void_p:
        if self._closed or not self.handle:
            raise ProjectorError("Projector handle is closed")
        return self.handle

    def _check_bool(self, ok: bool, action: str) -> None:
        if not ok:
            raise ProjectorError(f"{action} failed")

    @staticmethod
    def _range_value(name: str, value: int, minimum: int, maximum: int) -> int:
        result = int(value)
        if not minimum <= result <= maximum:
            raise ValueError(f"{name} must be in range {minimum}..{maximum}")
        return result

    def _write_command(
        self,
        command: str,
        read_reply: bool = True,
        delay_s: float = 0.1,
        size: int = 64,
    ) -> str:
        if read_reply:
            return self.write_read_text(command, delay_s=delay_s, size=size)
        self.write(command)
        return ""

    def led_on(self) -> None:
        self._check_bool(self.sdk.dll.TJSTPrjLedOn(self._require_open()), "LED on")

    def led_off(self) -> None:
        self._check_bool(self.sdk.dll.TJSTPrjLedOff(self._require_open()), "LED off")

    def project_black(self) -> None:
        self.set_mode(ProjectorMode.BLACK)

    def project_white(self) -> None:
        self.set_mode(ProjectorMode.WHITE)

    def project_cross(self) -> None:
        self.set_mode(ProjectorMode.CROSS)

    def project_chessboard(self) -> None:
        self.set_mode(ProjectorMode.CHESSBOARD)

    def set_mode(self, mode: int | ProjectorMode) -> None:
        value = int(ProjectorMode(int(mode)))
        self._check_bool(
            self.sdk.dll.TJSTPrjSetMode(self._require_open(), value),
            f"set mode {ProjectorMode(value).name}",
        )

    def set_color(self, color: int | ProjectorColor) -> None:
        value = int(ProjectorColor(int(color)))
        self._check_bool(
            self.sdk.dll.TJSTPrjSetColor(self._require_open(), value),
            f"set color {ProjectorColor(value).name}",
        )

    def set_light(self, light: int) -> None:
        value = int(light)
        if not 10 <= value <= 200:
            raise ValueError("light must be in range 10..200")
        self._check_bool(
            self.sdk.dll.TJSTPrjSetLight(self._require_open(), value),
            f"set light {value}",
        )

    def trigger_once(self, gray: int = 255) -> None:
        value = int(gray)
        if not 0 <= value <= 255:
            raise ValueError("gray must be in range 0..255")
        self._check_bool(
            self.sdk.dll.TJSTPrjTriggerOnce(self._require_open(), value),
            f"trigger once gray={value}",
        )

    def write(self, command: str | bytes, append_crlf: bool = True) -> int:
        if isinstance(command, str):
            data = command.encode("ascii")
        else:
            data = command
        if append_crlf and not data.endswith(b"\r\n"):
            data += b"\r\n"
        written = self.sdk.dll.TJSTPrjWrite(self._require_open(), data, len(data))
        if written <= 0:
            raise ProjectorError(f"write failed: {data!r}")
        return int(written)

    def write_read_text(
        self,
        command: str | bytes,
        delay_s: float = 0.1,
        size: int = 64,
        append_crlf: bool = True,
    ) -> str:
        self.write(command, append_crlf=append_crlf)
        return self.read_text(size=size, delay_s=delay_s)

    def select_builtin_image(
        self,
        image: int | BuiltInImage,
        read_reply: bool = True,
    ) -> str:
        """Select a projector built-in image with ``S0``..``S7``.

        ``S0``..``S3`` map to black, white, cross and chessboard. The TJ50
        manual also documents ``S4`` custom image 1 and optional ``S6``/``S7``
        cross variants. For the core four modes, ``set_mode()`` uses the DLL
        helper instead.
        """

        value = int(BuiltInImage(int(image)))
        return self._write_command(f"S{value}", read_reply=read_reply)

    def set_power_on_default_image(
        self,
        image: int | BuiltInImage,
        read_reply: bool = True,
    ) -> str:
        """Set the power-on default image with ``S8 n``."""

        value = int(BuiltInImage(int(image)))
        return self._write_command(f"S8 {value}", read_reply=read_reply)

    def set_image_rotation(
        self,
        rotation: int | ImageRotation,
        read_reply: bool = True,
    ) -> str:
        """Set image rotation with ``S9 n``.

        Values are 0 none, 1 X flip, 2 Y flip and 3 XY flip. The manual notes
        that the setting is saved in user flash and survives power cycles.
        """

        value = int(ImageRotation(int(rotation)))
        return self._write_command(f"S9 {value}", read_reply=read_reply)

    def set_download_frame_count(self, frame_count: int, read_reply: bool = True) -> str:
        """Set the number of custom pattern frames to download.

        This is the ``MB n`` command used by the vendor MFC demo before flash
        erase and per-byte pattern writes. TJ50 stores up to 100 user images.
        """

        count = self._range_value(
            "frame_count", frame_count, 1, MAX_CUSTOM_PATTERN_FRAMES
        )
        return self._write_command(f"MB {count}", read_reply=read_reply, delay_s=0.05)

    def erase_custom_patterns(self, tries: int = 3) -> str:
        """Erase custom pattern storage with the vendor demo's ``FE`` command."""

        self.write("FE")
        last_reply = ""
        for _ in range(int(tries)):
            last_reply = self.read_text(delay_s=1.0)
            if "F0" in last_reply:
                return last_reply
        raise ProjectorError(f"Projector did not confirm FE erase. Last reply: {last_reply!r}")

    def write_pattern_byte(self, position: int, value: int, read_reply: bool = False) -> str:
        """Write one custom pattern byte with ``FW{position} {value}``."""

        pos = int(position)
        val = int(value)
        if pos < 0:
            raise ValueError("position must be >= 0")
        if not 0 <= val <= 255:
            raise ValueError("value must be in range 0..255")
        command = f"FW{pos} {val}"
        return self._write_command(command, read_reply=read_reply, delay_s=0.2)

    def write_pattern_rgb_byte(
        self,
        position: int,
        red: int,
        green: int,
        blue: int,
        read_reply: bool = False,
    ) -> str:
        """Write one RGB custom pattern pixel with ``Fw pos r g b``."""

        pos = int(position)
        if pos < 0:
            raise ValueError("position must be >= 0")
        r = self._range_value("red", red, 0, 255)
        g = self._range_value("green", green, 0, 255)
        b = self._range_value("blue", blue, 0, 255)
        return self._write_command(
            f"Fw {pos} {r} {g} {b}",
            read_reply=read_reply,
            delay_s=0.2,
        )

    def configure_trigger_sequence(
        self,
        repeat_frames: int,
        image_count: int,
        trigger_start_frame: int = 0,
        image_start_index: int = 0,
        read_reply: bool = True,
    ) -> str:
        """Configure trigger playback with ``MA a b c d``.

        According to the TJ50 V1.5 manual:
        ``a`` is repeat frames per image (0..255), ``b`` is the number of
        images to trigger (0..100), ``c`` is whether triggering starts at frame
        0 or frame 1, and ``d`` is the user image start index. The manual's
        examples use ``0`` as the default start index, so this wrapper accepts
        0..100.
        """

        repeat = self._range_value("repeat_frames", repeat_frames, 0, 255)
        count = self._range_value("image_count", image_count, 0, MAX_CUSTOM_PATTERN_FRAMES)
        start_frame = self._range_value("trigger_start_frame", trigger_start_frame, 0, 1)
        start_index = self._range_value(
            "image_start_index", image_start_index, 0, MAX_CUSTOM_PATTERN_FRAMES
        )
        return self._write_command(
            f"MA {repeat} {count} {start_frame} {start_index}",
            read_reply=read_reply,
        )

    def configure_trigger_sequence_ma(
        self,
        start_frame: int,
        pattern_frame_count: int,
        trailing_dark_frames: int = 0,
        reserved: int = 0,
        read_reply: bool = True,
    ) -> str:
        """Compatibility wrapper for older code that used the raw ``MA`` order.

        Prefer ``configure_trigger_sequence()`` for the manual's parameter
        names. The positional order is unchanged: ``MA a b c d``.
        """

        return self.configure_trigger_sequence(
            repeat_frames=start_frame,
            image_count=pattern_frame_count,
            trigger_start_frame=trailing_dark_frames,
            image_start_index=reserved,
            read_reply=read_reply,
        )

    def save_trigger_settings(self, read_reply: bool = True) -> str:
        """Persist the current trigger parameters with ``MS``."""

        return self._write_command("MS", read_reply=read_reply)

    def set_horizontal_pattern_count(self, count: int, read_reply: bool = True) -> str:
        """Set the first ``count`` custom images as horizontal patterns via ``MD``.

        The manual says ``MD 0`` makes all images use the other orientation and
        ``MD n`` marks the first n images as horizontal patterns.
        """

        value = self._range_value("count", count, 0, MAX_CUSTOM_PATTERN_FRAMES)
        return self._write_command(f"MD {value}", read_reply=read_reply)

    def set_pattern_orientation_block(
        self,
        block: int,
        byte0: int,
        byte1: int,
        byte2: int,
        byte3: int,
        read_reply: bool = True,
    ) -> str:
        """Set 32 per-image orientation bits with ``MF block b0 b1 b2 b3``.

        ``block`` 0..3 covers images 0..31, 32..63, 64..95 and 96..127.
        The four byte arguments are the bitfield bytes described by the manual.
        """

        blk = self._range_value("block", block, 0, 3)
        values = [
            self._range_value("byte0", byte0, 0, 255),
            self._range_value("byte1", byte1, 0, 255),
            self._range_value("byte2", byte2, 0, 255),
            self._range_value("byte3", byte3, 0, 255),
        ]
        return self._write_command(
            f"MF {blk} {values[0]} {values[1]} {values[2]} {values[3]}",
            read_reply=read_reply,
        )

    def set_trigger_mode(
        self,
        mode: int | TriggerMode,
        read_reply: bool = True,
    ) -> str:
        """Select trigger mode with ``B n``."""

        value = int(TriggerMode(int(mode)))
        return self._write_command(f"B {value}", read_reply=read_reply)

    def set_oblique_direction(
        self,
        direction: int | ObliqueDirection,
        read_reply: bool = True,
    ) -> str:
        """Set oblique-pattern direction with ``MX n``."""

        value = int(ObliqueDirection(int(direction)))
        return self._write_command(f"MX {value}", read_reply=read_reply)

    def next_pattern(self, read_reply: bool = True) -> str:
        """Advance to the next pattern with ``N`` in single-frame mode."""

        return self._write_command("N", read_reply=read_reply)

    def set_light_source_mode(
        self,
        mode: int | LightSourceMode,
        read_reply: bool = True,
    ) -> str:
        """Select light-source/trigger-source mode with ``D n``."""

        value = int(LightSourceMode(int(mode)))
        return self._write_command(f"D {value}", read_reply=read_reply)

    def software_trigger(self, read_reply: bool = True) -> str:
        """Trigger one projection with the raw ``T`` command."""

        return self._write_command("T", read_reply=read_reply)

    def set_auto_light(self, light: int, read_reply: bool = True) -> str:
        """Set LED current/brightness with ``LA n`` (0..175)."""

        value = self._range_value("light", light, 0, 175)
        return self._write_command(f"LA {value}", read_reply=read_reply)

    def set_power_on_led(self, on: bool, read_reply: bool = True) -> str:
        """Set power-on LED state with ``LD``.

        ``True`` sends ``LD 0`` for power-on LED enabled. ``False`` sends
        ``LD 1`` for power-on LED disabled. The manual says this is saved
        automatically.
        """

        return self._write_command(f"LD {0 if on else 1}", read_reply=read_reply)

    def set_dlp_exposure_us(self, exposure_us: int, read_reply: bool = True) -> str:
        """Set one DLP frame exposure time with ``LP n`` in microseconds."""

        value = self._range_value("exposure_us", exposure_us, 2555, 7550)
        return self._write_command(f"LP {value}", read_reply=read_reply)

    def erase_user_data(self, read_reply: bool = True) -> str:
        """Erase user data with ``pe``."""

        return self._write_command("pe", read_reply=read_reply)

    def write_user_data(
        self,
        address: int,
        value: int,
        read_reply: bool = True,
    ) -> str:
        """Write one user data byte with ``pw address value``."""

        addr = self._range_value("address", address, 0, 200)
        val = self._range_value("value", value, 0, 255)
        return self._write_command(f"pw {addr} {val}", read_reply=read_reply)

    def read_user_data(self, address: int, read_reply: bool = True) -> str:
        """Read one user data byte with ``pr address``."""

        addr = self._range_value("address", address, 0, 200)
        return self._write_command(f"pr {addr}", read_reply=read_reply)

    def read_firmware_version(self) -> str:
        """Read firmware version with ``v``."""

        return self._write_command("v", read_reply=True)

    def read_led_temperature(self) -> str:
        """Read LED temperature sensor value with ``W``."""

        return self._write_command("W", read_reply=True)

    def send_hardware_trigger_pulse(self, gray: int = 255) -> None:
        """Send one projector trigger pulse command via ``TJSTPrjTriggerOnce``."""

        self.trigger_once(gray)

    def cmd_not_back(self) -> None:
        if not hasattr(self.sdk.dll, "TJSTPrjCmdNotBack"):
            raise ProjectorError("TJSTPrjCmdNotBack is not exported by this DLL")
        self.sdk.dll.TJSTPrjCmdNotBack(self._require_open())

    def read(self, size: int = 64, delay_s: float = 0.0) -> bytes:
        if delay_s > 0:
            time.sleep(delay_s)
        buf = ctypes.create_string_buffer(int(size))
        count = self.sdk.dll.TJSTPrjRead(self._require_open(), buf, int(size) - 1)
        if count <= 0:
            return b""
        return bytes(buf.raw[:count])

    def read_text(self, size: int = 64, delay_s: float = 0.0) -> str:
        return self.read(size=size, delay_s=delay_s).decode("ascii", errors="replace")

    def download_pattern_file(
        self,
        file_path: os.PathLike[str] | str,
        width: int = DEFAULT_PATTERN_WIDTH,
        ack_every: int = 256,
        write_delay_s: float = 0.02,
    ) -> None:
        """Download a raw stripe-pattern file using the Demo's MB/FE/FW flow.

        The file must contain one byte per projected pixel column. Its length
        must be an integer multiple of ``width``.
        """

        width_value = self._range_value("width", width, 1, 10000)
        data = Path(file_path).read_bytes()
        if not data or len(data) % width_value != 0:
            raise ValueError(
                f"Pattern file length must be a non-zero multiple of width={width_value}"
            )

        image_count = len(data) // width_value
        self.set_download_frame_count(image_count)
        self.erase_custom_patterns()

        for pos, value in enumerate(data):
            self.write_pattern_byte(pos, value)
            if write_delay_s > 0:
                time.sleep(write_delay_s)
            if ack_every > 0 and pos % ack_every == ack_every - 1:
                self.read_text(delay_s=0.2)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List TJST projector devices.")
    parser.add_argument("--dll", help="Path to TJSTProjectorApi.dll")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    sdk = TJSTProjectorSDK(args.dll)
    print(f"DLL: {sdk.dll_path}")
    devices = sdk.enum_devices()
    print(f"Found {len(devices)} projector(s)")
    for device in devices:
        print(device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
