# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import cam_projector
import real_camera
import real_projector
import structured_light_rebuild
from My_HKCamera import CameraError, My_HKCamera
from My_Projector import Projector
from Ui_BinocularCamera import Ui_MainWindow


PROJECT_ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = PROJECT_ROOT / "calib_pictures"
CALIB_LEFT_DIR_NAME = "Left"
CALIB_RIGHT_DIR_NAME = "Right"
DATA_DIR = PROJECT_ROOT / "data"
CALIB_DIR = PROJECT_ROOT / "config" / "data_calib"
DEFAULT_CALIB_FILE = CALIB_DIR / "calib.txt"
CALIB_PATH_CONFIG = CALIB_DIR / "calib_path.json"

DEFAULT_EXPOSURE_US = 8000
DEFAULT_GAIN = 0
DEFAULT_PROJECTOR_EXPOSURE_US = 8000
DEFAULT_PROJECTOR_BRIGHTNESS = 40
PROJECTOR_WARMUP_START_FRAME = 0
PROJECTOR_WARMUP_PATTERN_COUNT = 12
PREVIEW_TIMEOUT_MS = 50
PREVIEW_INTERVAL_MS = 30

STRIPE_OPTIONS = (
    (0, 36),   # 三频十二步: 3 * 12
    (36, 18),  # 三频六步:   3 * 6
    (54, 12),  # 三频四步:   3 * 4
    (66, 9),   # 三频三步:   3 * 3
    (75, 9),   # 互补格雷码: 4 phase + 4 gray + 1 complementary gray
    (84, 8),   # 双频四步:   2 * 4
)


def set_projector_exposure_us(projector: Projector, exposure_us: int) -> None:
    value = int(exposure_us)
    try:
        projector.set_dlp_exposure_us(value)
    except ValueError:
        projector.write("LP %d" % value)


def format_named_matrix(name: str, matrix: np.ndarray) -> str:
    rows = []
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    for row in values:
        rows.append("\t".join("%38.32f" % float(value) for value in row))
    return "%s\n%s\n" % (name, "\n".join(rows))


def projector_sequence_wait_s(pattern_count: int) -> float:
    if real_projector.PROJECTOR_FPS <= 0:
        return 0.0
    return (
        (int(real_projector.REPEAT_FRAMES) + 1)
        * int(pattern_count)
        / float(real_projector.PROJECTOR_FPS)
    )


class ScanWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        left_camera: My_HKCamera,
        right_camera: My_HKCamera,
        projector: Projector,
        exposure_us: int,
        gain: int,
        projector_exposure_us: int,
        stripe_index: int,
        calib_path: Path,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.projector = projector
        self.exposure_us = exposure_us
        self.gain = gain
        self.projector_exposure_us = projector_exposure_us
        self.stripe_index = stripe_index
        self.calib_path = calib_path

    def run(self) -> None:
        try:
            capture_dir, left_save_dir, right_save_dir = cam_projector.create_next_capture_dirs()
            self._prepare_scan_cameras()
            cam_projector.set_projector_brightness(self.projector)
            set_projector_exposure_us(self.projector, self.projector_exposure_us)

            if cam_projector.CLEAR_BUFFER_BEFORE_TRIGGER:
                cam_projector.clear_stereo_buffers(self.left_camera, self.right_camera)

            cam_projector.trigger_burned_pattern_sequence(self.projector)
            sequence_start = time.monotonic()

            if cam_projector.PROJECTOR_START_DELAY_S > 0:
                time.sleep(cam_projector.PROJECTOR_START_DELAY_S)
            if cam_projector.DISCARD_LEADING_FRAMES > 0:
                cam_projector.discard_stereo_frames(
                    self.left_camera,
                    self.right_camera,
                    cam_projector.DISCARD_LEADING_FRAMES,
                )

            left_frames, right_frames = cam_projector.acquire_stereo_frames(
                self.left_camera,
                self.right_camera,
                cam_projector.CAPTURE_PATTERN_COUNT,
            )

            cam_projector.stop_camera_grabbing(self.left_camera)
            cam_projector.stop_camera_grabbing(self.right_camera)
            cam_projector.save_stereo_frames(
                left_frames,
                right_frames,
                left_save_dir,
                right_save_dir,
            )

            remaining_wait_s = cam_projector.PROJECTION_WAIT_S - (
                time.monotonic() - sequence_start
            )
            if remaining_wait_s > 0:
                time.sleep(remaining_wait_s)

            try:
                self.projector.project_white()
            except Exception:
                pass

            structured_light_rebuild.reconstruct_capture(
                capture_dir,
                self.stripe_index,
                calib_path=self.calib_path,
            )

            self.finished_ok.emit(str(capture_dir))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._restore_live_cameras()

    def _prepare_scan_cameras(self) -> None:
        for camera in (self.left_camera, self.right_camera):
            cam_projector.stop_camera_grabbing(camera)
            camera.set_exposure_time(self.exposure_us)
            camera.set_gain(self.gain)
            camera.set_trigger_mode(True, source=cam_projector.CAMERA_TRIGGER_SOURCE)
            cam_projector.set_image_node_num(camera, cam_projector.CAMERA_IMAGE_NODE_NUM)
            camera.start_grabbing()

        if cam_projector.CAMERA_READY_DELAY_S > 0:
            time.sleep(cam_projector.CAMERA_READY_DELAY_S)

    def _restore_live_cameras(self) -> None:
        for camera in (self.left_camera, self.right_camera):
            try:
                cam_projector.stop_camera_grabbing(camera)
                camera.set_trigger_mode(False)
                camera.set_exposure_time(self.exposure_us)
                camera.set_gain(self.gain)
                camera.start_grabbing()
            except Exception:
                pass


class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.left_camera: Optional[My_HKCamera] = None
        self.right_camera: Optional[My_HKCamera] = None
        self.projector: Optional[Projector] = None
        self.projector_enabled = False
        self.scan_worker: Optional[ScanWorker] = None
        self.left_camera_index = int(cam_projector.LEFT_CAMERA_INDEX)
        self.right_camera_index = int(cam_projector.RIGHT_CAMERA_INDEX)
        self.last_left_frame = None
        self.last_right_frame = None
        self.last_scan_dir: Optional[Path] = None
        self.zoom = {
            self.Left_show: 1.0,
            self.Right_show: 1.0,
        }
        self.image_top_left = {
            self.Left_show: None,
            self.Right_show: None,
        }

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(PREVIEW_INTERVAL_MS)
        self.preview_timer.timeout.connect(self.update_preview)

        self.setup_controls()
        self.connect_signals()
        self.set_warning("idle")

    def setup_controls(self) -> None:
        self.setWindowTitle("Binocular Camera")

        self.Open.setText("打开设备")
        self.Disconnet.setText("断开设备")
        self.close_pro.setText("关闭投影仪")
        self.label_2.setText("曝光:")
        self.label_5.setText("增益:")
        self.Crosshair.setText("十字线")
        self.label_3.setText("测试图案:")
        self.label_4.setText("投影亮度:")
        self.label_6.setText("条纹类型:")
        self.exchange_camera.setText("调转相机")
        self.Scan_rebuild.setText("扫描重建")
        self.photo.setText("单帧拍照")
        self.Calibration.setText("双目标定")
        self.open_calib.setText("选择标定文件")
        self.open_folder_1.setText("打开左条纹目录")
        self.open_folder_2.setText("打开右条纹目录")
        self.open_folder.setText("打开点云目录")

        self.choose_pro.clear()
        self.choose_pro.addItems(["全白", "棋盘格", "中心线"])
        self.Change_stripe.clear()
        # self.Change_stripe.addItems(["三频六步", "三频十二步", "互补格雷码", "三频三步"])
        self.Change_stripe.addItems(["三频十二步", "三频三步", "互补格雷码", "双频互补"])

        self.Change_stripe.clear()
        self.Change_stripe.addItems([
            "三频十二步",
            "三频六步",
            "三频四步",
            "三频三步",
            "互补格雷码",
            "双频互补",
        ])

        self.change_exposure.setRange(1, 1_000_000)
        self.change_exposure.setSingleStep(1000)
        self.change_exposure.setValue(DEFAULT_EXPOSURE_US)
        self.change_exposure.setSuffix(" us")
        self.change_exposure.setMinimumWidth(95)

        self.change_gain.setRange(0, 48)
        self.change_gain.setSingleStep(1)
        self.change_gain.setValue(DEFAULT_GAIN)
        self.change_gain.setMinimumWidth(70)

        self.Change_brightness.setRange(10, 200)
        self.Change_brightness.setSingleStep(5)
        self.Change_brightness.setValue(DEFAULT_PROJECTOR_BRIGHTNESS)
        self.Change_brightness.setMinimumWidth(45)

        for label in (self.Left_show, self.Right_show):
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMouseTracking(True)
            label.installEventFilter(self)
            label.setText("未连接")

    def connect_signals(self) -> None:
        self.Open.clicked.connect(self.open_devices)
        self.Disconnet.clicked.connect(self.disconnect_devices)
        self.close_pro.clicked.connect(self.toggle_projector_output)
        self.change_exposure.valueChanged.connect(self.change_exposure_live)
        self.change_gain.valueChanged.connect(self.change_gain_live)
        self.Crosshair.stateChanged.connect(lambda _state: self.refresh_last_frames())
        self.choose_pro.currentIndexChanged.connect(self.project_test_pattern)
        self.Change_brightness.valueChanged.connect(self.change_brightness_live)
        self.Change_stripe.currentIndexChanged.connect(self.apply_stripe_settings)
        self.exchange_camera.clicked.connect(self.exchange_cameras)
        self.Scan_rebuild.clicked.connect(self.scan_rebuild)
        self.photo.clicked.connect(self.capture_photo)
        self.Calibration.clicked.connect(self.run_stereo_calibration)
        self.open_calib.clicked.connect(self.select_calibration_file)
        self.open_folder_1.clicked.connect(self.open_left_stripe_folder)
        self.open_folder_2.clicked.connect(self.open_right_stripe_folder)
        self.open_folder.clicked.connect(self.open_capture_folder)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched in self.zoom and event.type() == QtCore.QEvent.Wheel:
            image = self.current_frame_for_label(watched)
            if image is None:
                return True

            old_zoom = self.zoom[watched]
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.2, min(5.0, old_zoom * factor))

            if new_zoom != old_zoom:
                old_size = self.scaled_image_size(watched, image, old_zoom)
                old_top_left = self.image_top_left[watched]
                if old_top_left is None:
                    old_top_left = self.default_top_left(watched, old_size)

                mouse_pos = QtCore.QPointF(event.pos())
                ratio = new_zoom / old_zoom
                new_top_left = mouse_pos - (mouse_pos - old_top_left) * ratio
                new_size = self.scaled_image_size(watched, image, new_zoom)

                self.zoom[watched] = new_zoom
                self.image_top_left[watched] = self.clamp_top_left(
                    watched,
                    new_top_left,
                    new_size,
                )

            self.refresh_last_frames()
            return True
        return super().eventFilter(watched, event)

    def open_devices(self) -> None:
        if self.devices_open:
            self.statusbar.showMessage("设备已经打开。", 3000)
            return

        try:
            self.apply_runtime_settings()
            self.sync_camera_indices()
            self.left_camera = My_HKCamera(device_index=self.left_camera_index)
            self.right_camera = My_HKCamera(device_index=self.right_camera_index)
            self.left_camera.open()
            self.right_camera.open()
            self.left_camera.set_exposure_time(self.exposure_us)
            self.right_camera.set_exposure_time(self.exposure_us)
            self.left_camera.set_gain(self.gain)
            self.right_camera.set_gain(self.gain)
            self.left_camera.start_grabbing()
            self.right_camera.start_grabbing()

            self.projector = real_projector.open_projector(cam_projector.DEVICE_INDEX)
            set_projector_exposure_us(self.projector, DEFAULT_PROJECTOR_EXPOSURE_US)
            self.projector.set_light(self.projector_brightness)
            self.projector_enabled = True
            self.close_pro.setText("关闭投影仪")
            self.warm_up_projector()
            self.projector.project_white()

            self.preview_timer.start()
            self.set_warning("ready")
            self.statusbar.showMessage("相机和投影仪已打开。", 3000)
        except Exception as exc:
            self.disconnect_devices(show_message=False)
            self.set_warning("idle")
            QtWidgets.QMessageBox.critical(self, "打开设备失败", str(exc))

    def disconnect_devices(self, show_message: bool = True) -> None:
        if self.scan_worker is not None and self.scan_worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "正在扫描", "扫描过程中暂不能断开设备。")
            return

        self.preview_timer.stop()
        self._close_camera(self.left_camera)
        self._close_camera(self.right_camera)
        self.left_camera = None
        self.right_camera = None

        if self.projector is not None:
            try:
                real_projector.close_projector(self.projector)
            except Exception:
                pass
            self.projector = None
            self.projector_enabled = False
            self.close_pro.setText("关闭投影仪")

        self.last_left_frame = None
        self.last_right_frame = None
        self.Left_show.clear()
        self.Right_show.clear()
        self.Left_show.setText("未连接")
        self.Right_show.setText("未连接")
        self.image_top_left[self.Left_show] = None
        self.image_top_left[self.Right_show] = None
        self.set_warning("idle")

        if show_message:
            self.statusbar.showMessage("设备已断开。", 3000)

    def toggle_projector_output(self) -> None:
        if self.projector is None:
            QtWidgets.QMessageBox.warning(self, "投影仪未打开", "请先点击“打开设备”。")
            return
        try:
            self.set_projector_output_enabled(not self.projector_enabled)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "投影仪切换失败", str(exc))

    def set_projector_output_enabled(self, enabled: bool) -> None:
        if self.projector is None:
            raise RuntimeError("投影仪未打开。")
        if enabled:
            self.projector.led_on()
            self.projector.set_light(self.projector_brightness)
            self.projector.project_white()
            self.projector_enabled = True
            self.close_pro.setText("关闭投影仪")
            self.statusbar.showMessage("投影仪已打开。", 3000)
        else:
            self.projector.led_off()
            self.projector_enabled = False
            self.close_pro.setText("打开投影仪")
            self.statusbar.showMessage("投影仪已关闭。", 3000)

    def warm_up_projector(self) -> None:
        if self.projector is None:
            return
        self.projector.led_on()
        self.projector.configure_trigger_sequence(
            repeat_frames=real_projector.REPEAT_FRAMES,
            image_count=PROJECTOR_WARMUP_PATTERN_COUNT,
            trigger_start_frame=real_projector.TRIGGER_START_FRAME,
            image_start_index=PROJECTOR_WARMUP_START_FRAME,
            read_reply=False,
        )
        self.projector.send_hardware_trigger_pulse(255)
        time.sleep(projector_sequence_wait_s(PROJECTOR_WARMUP_PATTERN_COUNT))

    def exchange_cameras(self) -> None:
        if self.scan_worker is not None and self.scan_worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "正在扫描", "扫描过程中暂不能调转相机。")
            return

        self.left_camera_index, self.right_camera_index = (
            self.right_camera_index,
            self.left_camera_index,
        )
        self.sync_camera_indices()

        self.left_camera, self.right_camera = self.right_camera, self.left_camera
        self.last_left_frame, self.last_right_frame = (
            self.last_right_frame,
            self.last_left_frame,
        )
        self.image_top_left[self.Left_show] = None
        self.image_top_left[self.Right_show] = None
        self.refresh_last_frames()

        self.statusbar.showMessage(
            "左右相机已调转: Left index=%d, Right index=%d"
            % (self.left_camera_index, self.right_camera_index),
            4000,
        )

    def update_preview(self) -> None:
        if not self.devices_open:
            return

        try:
            left_image, _left_info = self.left_camera.get_frame(
                timeout_ms=PREVIEW_TIMEOUT_MS,
                output="mono",
            )
            right_image, _right_info = self.right_camera.get_frame(
                timeout_ms=PREVIEW_TIMEOUT_MS,
                output="mono",
            )
        except CameraError as exc:
            self.statusbar.showMessage("预览取帧失败: %s" % exc, 1000)
            return

        self.last_left_frame = left_image
        self.last_right_frame = right_image
        self.show_image(self.Left_show, left_image)
        self.show_image(self.Right_show, right_image)

    def change_exposure_live(self, value: int) -> None:
        self.apply_runtime_settings()
        for camera in (self.left_camera, self.right_camera):
            if camera is not None and camera.is_open:
                try:
                    camera.set_exposure_time(value)
                except Exception as exc:
                    self.statusbar.showMessage("曝光设置失败: %s" % exc, 3000)

    def change_gain_live(self, value: int) -> None:
        for camera in (self.left_camera, self.right_camera):
            if camera is not None and camera.is_open:
                try:
                    camera.set_gain(value)
                except Exception as exc:
                    self.statusbar.showMessage("增益设置失败: %s" % exc, 3000)

    def change_brightness_live(self, value: int) -> None:
        self.apply_runtime_settings()
        if self.projector is not None:
            try:
                self.projector.set_light(value)
                self.statusbar.showMessage("投影亮度已更新为 %d。" % value, 2000)
            except Exception as exc:
                self.statusbar.showMessage("亮度设置失败: %s" % exc, 3000)

    def project_test_pattern(self, index: int) -> None:
        if self.projector is None:
            return

        try:
            if not self.projector_enabled:
                self.set_projector_output_enabled(True)
            if index == 0:
                self.projector.project_white()
            elif index == 1:
                real_projector.project_chessboard(self.projector)
            elif index == 2:
                real_projector.project_cross(self.projector)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "投影失败", str(exc))

    def apply_stripe_settings(self, _index: Optional[int] = None) -> None:
        self.apply_runtime_settings()
        start_frame, pattern_count = STRIPE_OPTIONS[self.Change_stripe.currentIndex()]

        cam_projector.START_FRAME = start_frame
        cam_projector.BURNED_PATTERN_COUNT = pattern_count
        cam_projector.CAPTURE_PATTERN_COUNT = pattern_count + 1
        cam_projector.CAMERA_IMAGE_NODE_NUM = cam_projector.CAPTURE_PATTERN_COUNT + 4

        real_projector.START_FRAME = start_frame
        real_projector.BURNED_PATTERN_COUNT = pattern_count

        cam_projector.PROJECTION_WAIT_S = real_projector._burned_pattern_wait_s() * (
            cam_projector.CAPTURE_PATTERN_COUNT / pattern_count
        )
        self.statusbar.showMessage(
            "条纹参数: START_FRAME=%d, BURNED_PATTERN_COUNT=%d"
            % (start_frame, pattern_count),
            3000,
        )

    def scan_rebuild(self) -> None:
        if not self.devices_open:
            QtWidgets.QMessageBox.warning(self, "设备未打开", "请先点击“打开设备”。")
            return
        if self.scan_worker is not None and self.scan_worker.isRunning():
            return

        if not self.projector_enabled:
            try:
                self.set_projector_output_enabled(True)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "投影仪打开失败", str(exc))
                return

        self.apply_stripe_settings()
        self.preview_timer.stop()
        self.set_warning("scan")
        self.set_scan_buttons_enabled(False)
        self.statusbar.showMessage("正在扫描采集条纹图并重建...")

        self.scan_worker = ScanWorker(
            self.left_camera,
            self.right_camera,
            self.projector,
            self.exposure_us,
            self.gain,
            DEFAULT_PROJECTOR_EXPOSURE_US,
            self.Change_stripe.currentIndex(),
            self.current_calibration_path(),
            self,
        )
        self.scan_worker.finished_ok.connect(self.scan_finished)
        self.scan_worker.failed.connect(self.scan_failed)
        self.scan_worker.finished.connect(self.scan_thread_finished)
        self.scan_worker.start()

    def scan_finished(self, capture_dir: str) -> None:
        self.last_scan_dir = Path(capture_dir)
        self.statusbar.showMessage("扫描和重建完成，已保存到 %s" % capture_dir, 5000)

    def scan_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "扫描失败", message)

    def scan_thread_finished(self) -> None:
        self.set_scan_buttons_enabled(True)
        if self.devices_open:
            self.preview_timer.start()
            self.set_warning("ready")
        else:
            self.set_warning("idle")
        self.scan_worker = None

    def capture_photo(self) -> None:
        if self.last_left_frame is None or self.last_right_frame is None:
            QtWidgets.QMessageBox.warning(self, "没有图像", "请先打开设备并等待预览画面。")
            return

        left_dir = CAPTURE_DIR / CALIB_LEFT_DIR_NAME
        right_dir = CAPTURE_DIR / CALIB_RIGHT_DIR_NAME
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        left_path = left_dir / ("%s.bmp" % timestamp)
        right_path = right_dir / ("%s.bmp" % timestamp)

        ok_left = cv2.imwrite(str(left_path), self.last_left_frame)
        ok_right = cv2.imwrite(str(right_path), self.last_right_frame)
        if not ok_left or not ok_right:
            QtWidgets.QMessageBox.critical(self, "保存失败", "单帧图像保存失败。")
            return

        self.statusbar.showMessage("单帧照片已保存到 calib_pictures。", 4000)

    def run_stereo_calibration(self) -> None:
        left_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择左相机标定照片",
            str(CAPTURE_DIR / CALIB_LEFT_DIR_NAME),
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not left_paths:
            return

        right_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择右相机标定照片",
            str(CAPTURE_DIR / CALIB_RIGHT_DIR_NAME),
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if not right_paths:
            return

        board_text, ok = QtWidgets.QInputDialog.getText(
            self,
            "棋盘格内角点",
            "请输入内角点列数,行数：",
            text="11,8",
        )
        if not ok:
            return
        try:
            board_cols, board_rows = self.parse_board_size(board_text)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        square_size, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "棋盘格方格边长",
            "请输入方格边长：",
            value=1.0,
            min=0.000001,
            decimals=6,
        )
        if not ok:
            return

        save_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "保存标定文件",
            "请输入保存文件名：",
            text="calib.txt",
        )
        if not ok or not save_name.strip():
            return

        try:
            output_path = self.safe_calib_output_path(save_name)
            result = self.calibrate_stereo_from_images(
                [Path(path) for path in sorted(left_paths)],
                [Path(path) for path in sorted(right_paths)],
                (board_cols, board_rows),
                float(square_size),
            )
            self.write_calibration_file(output_path, result)
            self.write_calibration_path(output_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "标定失败", str(exc))
            return

        self.statusbar.showMessage("双目标定完成，已保存到 %s" % output_path, 5000)

    def select_calibration_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择标定文件",
            str(CALIB_DIR),
            "Calibration (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            calib_path = Path(path).resolve()
            if not calib_path.exists():
                raise FileNotFoundError(str(calib_path))
            self.write_calibration_path(calib_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "选择标定文件失败", str(exc))
            return
        self.statusbar.showMessage("当前标定文件: %s" % calib_path, 5000)

    def open_capture_folder(self) -> None:
        folder = self.rebuild_folder_for_open()
        if folder is None:
            QtWidgets.QMessageBox.warning(self, "没有重建目录", "请先完成一次扫描重建。")
            return
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, "没有重建目录", "目录不存在: %s" % folder)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def open_left_stripe_folder(self) -> None:
        self.open_stripe_folder(cam_projector.LEFT_PICTURE_DIR_NAME, "左相机条纹目录")

    def open_right_stripe_folder(self) -> None:
        self.open_stripe_folder(cam_projector.RIGHT_PICTURE_DIR_NAME, "右相机条纹目录")

    def open_stripe_folder(self, folder_name: str, title: str) -> None:
        scan_dir = self.last_scan_dir if self.last_scan_dir is not None else self.latest_data_dir()
        if scan_dir is None:
            QtWidgets.QMessageBox.warning(self, "没有扫描目录", "请先点击扫描重建并完成采集。")
            return
        folder = scan_dir / folder_name
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, title, "目录不存在: %s" % folder)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def rebuild_folder_for_open(self) -> Optional[Path]:
        scan_dir = self.last_scan_dir if self.last_scan_dir is not None else self.latest_data_dir()
        if scan_dir is None:
            return None
        return scan_dir / "rebuild"

    def current_calibration_path(self) -> Path:
        try:
            if CALIB_PATH_CONFIG.exists() and CALIB_PATH_CONFIG.stat().st_size > 0:
                with CALIB_PATH_CONFIG.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                path_text = data.get("calib_path") or data.get("path")
                if path_text:
                    calib_path = Path(path_text)
                    if not calib_path.is_absolute():
                        calib_path = PROJECT_ROOT / calib_path
                    if calib_path.exists():
                        return calib_path
        except Exception:
            pass
        return DEFAULT_CALIB_FILE

    @staticmethod
    def write_calibration_path(calib_path: Path) -> None:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        with CALIB_PATH_CONFIG.open("w", encoding="utf-8") as file:
            json.dump({"calib_path": str(calib_path.resolve())}, file, ensure_ascii=False, indent=2)

    @staticmethod
    def parse_board_size(text: str) -> tuple[int, int]:
        normalized = text.replace("，", ",").replace("x", ",").replace("X", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("请输入类似 11,8 的内角点列数和行数。")
        cols, rows = int(parts[0]), int(parts[1])
        if cols < 2 or rows < 2:
            raise ValueError("内角点列数和行数都必须大于 1。")
        return cols, rows

    @staticmethod
    def safe_calib_output_path(save_name: str) -> Path:
        name = Path(save_name.strip()).name
        if not name:
            raise ValueError("保存文件名不能为空。")
        path = CALIB_DIR / name
        if path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        return path

    def calibrate_stereo_from_images(
        self,
        left_paths: Sequence[Path],
        right_paths: Sequence[Path],
        board_size: tuple[int, int],
        square_size: float,
    ) -> dict[str, np.ndarray | float | tuple[int, int]]:
        if len(left_paths) != len(right_paths):
            raise RuntimeError("左右相机照片数量不一致。")
        if len(left_paths) < 3:
            raise RuntimeError("至少需要 3 组左右标定照片。")
        left_names = [path.name for path in left_paths]
        right_names = [path.name for path in right_paths]
        if left_names != right_names:
            raise RuntimeError("左右相机照片文件名不一致，请选择同名、同数量的左右照片。")

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )
        object_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        object_template[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
        object_template *= float(square_size)

        object_points = []
        left_points = []
        right_points = []
        image_size: Optional[tuple[int, int]] = None
        failed_pairs = []

        for left_path, right_path in zip(left_paths, right_paths):
            left_image = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
            if left_image is None or right_image is None:
                failed_pairs.append("%s / %s" % (left_path.name, right_path.name))
                continue
            if left_image.shape != right_image.shape:
                raise RuntimeError("左右照片尺寸不一致: %s / %s" % (left_path.name, right_path.name))
            if image_size is None:
                image_size = (left_image.shape[1], left_image.shape[0])

            found_left, corners_left = cv2.findChessboardCorners(left_image, board_size)
            found_right, corners_right = cv2.findChessboardCorners(right_image, board_size)
            if not found_left or not found_right:
                failed_pairs.append("%s / %s" % (left_path.name, right_path.name))
                continue

            corners_left = cv2.cornerSubPix(left_image, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_image, corners_right, (11, 11), (-1, -1), criteria)
            object_points.append(object_template.copy())
            left_points.append(corners_left)
            right_points.append(corners_right)

        if image_size is None:
            raise RuntimeError("没有读取到有效照片。")
        if len(object_points) < 3:
            raise RuntimeError(
                "有效角点照片少于 3 组，无法标定。未识别: %s"
                % ", ".join(failed_pairs[:8])
            )

        _ret_l, camera_l, dist_l, _rvecs_l, _tvecs_l = cv2.calibrateCamera(
            object_points,
            left_points,
            image_size,
            None,
            None,
        )
        _ret_r, camera_r, dist_r, _rvecs_r, _tvecs_r = cv2.calibrateCamera(
            object_points,
            right_points,
            image_size,
            None,
            None,
        )
        rms, camera_l, dist_l, camera_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            left_points,
            right_points,
            camera_l,
            dist_l,
            camera_r,
            dist_r,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        return {
            "image_size": image_size,
            "K1": camera_l,
            "K2": camera_r,
            "dist1": dist_l.reshape(-1),
            "dist2": dist_r.reshape(-1),
            "R": R,
            "T": T.reshape(3),
            "E": E,
            "F": F,
            "error": float(rms),
            "valid_pairs": len(object_points),
        }

    @staticmethod
    def write_calibration_file(path: Path, result: dict[str, np.ndarray | float | tuple[int, int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image_width, image_height = result["image_size"]
        dist1 = np.asarray(result["dist1"], dtype=np.float64).reshape(-1)
        dist2 = np.asarray(result["dist2"], dtype=np.float64).reshape(-1)
        radial_l = [dist1[0], dist1[1] if dist1.size > 1 else 0.0]
        radial_r = [dist2[0], dist2[1] if dist2.size > 1 else 0.0]
        tangent_l = [dist1[2] if dist1.size > 2 else 0.0, dist1[3] if dist1.size > 3 else 0.0]
        tangent_r = [dist2[2] if dist2.size > 2 else 0.0, dist2[3] if dist2.size > 3 else 0.0]

        parts = [
            format_named_matrix("imageSize", np.array([[image_height, image_width]], dtype=np.float64)),
            format_named_matrix("KK_L", np.asarray(result["K1"], dtype=np.float64).T),
            format_named_matrix("KK_R", np.asarray(result["K2"], dtype=np.float64).T),
            format_named_matrix("RadialDistortion_L", np.array([radial_l], dtype=np.float64)),
            format_named_matrix("RadialDistortion_R", np.array([radial_r], dtype=np.float64)),
            format_named_matrix("TangentialDistortion_L", np.array([tangent_l], dtype=np.float64)),
            format_named_matrix("TangentialDistortion_R", np.array([tangent_r], dtype=np.float64)),
            format_named_matrix("R", np.asarray(result["R"], dtype=np.float64)),
            format_named_matrix("T", np.asarray(result["T"], dtype=np.float64).reshape(1, 3)),
            format_named_matrix("E", np.asarray(result["E"], dtype=np.float64)),
            format_named_matrix("F", np.asarray(result["F"], dtype=np.float64)),
            "error\n%.8g\n" % float(result["error"]),
        ]
        path.write_text("\n".join(parts), encoding="utf-8")

    def show_image(self, label: QtWidgets.QLabel, image) -> None:
        qimage = self.numpy_to_qimage(image)
        source = QtGui.QPixmap.fromImage(qimage)
        canvas = QtGui.QPixmap(label.size())
        canvas.fill(QtGui.QColor(207, 207, 207))

        if not source.isNull():
            base_scale = min(
                label.width() / source.width(),
                label.height() / source.height(),
            )
            scale = base_scale * self.zoom[label]
            target_size = QtCore.QSize(
                max(1, int(source.width() * scale)),
                max(1, int(source.height() * scale)),
            )
            scaled = source.scaled(
                target_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            top_left = self.image_top_left[label]
            if top_left is None:
                top_left = self.default_top_left(label, scaled.size())
            top_left = self.clamp_top_left(label, top_left, scaled.size())
            self.image_top_left[label] = top_left
            x = int(top_left.x())
            y = int(top_left.y())

            painter = QtGui.QPainter(canvas)
            painter.drawPixmap(x, y, scaled)
            if self.Crosshair.isChecked():
                pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 1)
                painter.setPen(pen)
                center_x = label.width() // 2
                center_y = label.height() // 2
                painter.drawLine(center_x, 0, center_x, label.height())
                painter.drawLine(0, center_y, label.width(), center_y)
            painter.end()

        label.setPixmap(canvas)

    def refresh_last_frames(self) -> None:
        if self.last_left_frame is not None:
            self.show_image(self.Left_show, self.last_left_frame)
        if self.last_right_frame is not None:
            self.show_image(self.Right_show, self.last_right_frame)

    def current_frame_for_label(self, label: QtWidgets.QLabel):
        if label is self.Left_show:
            return self.last_left_frame
        if label is self.Right_show:
            return self.last_right_frame
        return None

    def scaled_image_size(
        self,
        label: QtWidgets.QLabel,
        image,
        zoom: float,
    ) -> QtCore.QSize:
        height, width = image.shape[:2]
        base_scale = min(label.width() / width, label.height() / height)
        return QtCore.QSize(
            max(1, int(width * base_scale * zoom)),
            max(1, int(height * base_scale * zoom)),
        )

    @staticmethod
    def default_top_left(
        label: QtWidgets.QLabel,
        image_size: QtCore.QSize,
    ) -> QtCore.QPointF:
        return QtCore.QPointF(
            (label.width() - image_size.width()) / 2,
            (label.height() - image_size.height()) / 2,
        )

    def clamp_top_left(
        self,
        label: QtWidgets.QLabel,
        top_left: QtCore.QPointF,
        image_size: QtCore.QSize,
    ) -> QtCore.QPointF:
        if image_size.width() <= label.width():
            x = (label.width() - image_size.width()) / 2
        else:
            x = min(0.0, max(label.width() - image_size.width(), top_left.x()))

        if image_size.height() <= label.height():
            y = (label.height() - image_size.height()) / 2
        else:
            y = min(0.0, max(label.height() - image_size.height(), top_left.y()))

        return QtCore.QPointF(x, y)

    def apply_runtime_settings(self) -> None:
        cam_projector.EXPOSURE_TIME_US = float(self.exposure_us)
        real_camera.EXPOSURE_TIME_US = float(self.exposure_us)
        cam_projector.PROJECTOR_BRIGHTNESS = int(self.projector_brightness)
        real_projector._BRIGHTNESS = int(self.projector_brightness)
        self.sync_camera_indices()

    def sync_camera_indices(self) -> None:
        cam_projector.LEFT_CAMERA_INDEX = int(self.left_camera_index)
        cam_projector.RIGHT_CAMERA_INDEX = int(self.right_camera_index)

    def set_warning(self, state: str) -> None:
        colors = {
            "idle": "rgb(255, 255, 0)",
            "ready": "rgb(0, 190, 0)",
            "scan": "rgb(220, 0, 0)",
        }
        self.Warning.setStyleSheet("background-color: %s;" % colors[state])

    def set_scan_buttons_enabled(self, enabled: bool) -> None:
        for widget in (
            self.Open,
            self.Disconnet,
            self.close_pro,
            self.change_exposure,
            self.change_gain,
            self.choose_pro,
            self.Change_brightness,
            self.Change_stripe,
            self.exchange_camera,
            self.Scan_rebuild,
            self.photo,
            self.Calibration,
            self.open_calib,
            self.open_folder_1,
            self.open_folder_2,
            self.open_folder,
        ):
            widget.setEnabled(enabled)

    def latest_data_dir(self) -> Optional[Path]:
        if not DATA_DIR.exists():
            return None
        numbered_dirs = [
            path
            for path in DATA_DIR.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        if not numbered_dirs:
            return None
        return max(numbered_dirs, key=lambda path: int(path.name))

    @staticmethod
    def _close_camera(camera: Optional[My_HKCamera]) -> None:
        if camera is None:
            return
        try:
            camera.close()
        except Exception:
            pass

    @staticmethod
    def numpy_to_qimage(image) -> QtGui.QImage:
        if image.ndim == 2:
            height, width = image.shape
            bytes_per_line = image.strides[0]
            return QtGui.QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_Grayscale8,
            ).copy()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        return QtGui.QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        ).copy()

    @property
    def devices_open(self) -> bool:
        return (
            self.left_camera is not None
            and self.right_camera is not None
            and self.projector is not None
        )

    @property
    def exposure_us(self) -> int:
        return int(self.change_exposure.value())

    @property
    def gain(self) -> int:
        return int(self.change_gain.value())

    @property
    def projector_brightness(self) -> int:
        return int(self.Change_brightness.value())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.scan_worker is not None and self.scan_worker.isRunning():
            self.scan_worker.wait(5000)
        self.disconnect_devices(show_message=False)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
