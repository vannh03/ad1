# ui/dashboard.py
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue, Event
import cv2
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QProgressBar, QTextEdit, QFrame, QSplitter, QCheckBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon

from configs.settings import MODEL_DETECTION_PATH, PROJECT_ROOT
from core import (
    video_reader_process,
    detector_tracker_process,
    feature_mlp_rule_process
)


class RealtimeFPSTracker:
    def __init__(self, window_size=20):
        self.timestamps = []
        self.window_size = window_size

    def update(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / duration

class PipelineWorker(QThread):
    frame_ready = pyqtSignal(QImage, dict)
    log_message = pyqtSignal(str)
    pipeline_finished = pyqtSignal()

    def __init__(self, source, weights, imgsz=(320, 320), conf_thres=0.25, iou_thres=0.45, device='cpu', save_video=False):
        super().__init__()
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.save_video = save_video

        self._is_running = True
        self._is_paused = False
        self.stop_event = Event()
        self.processes = []
        self.vid_writer = None
        self.fps_tracker = RealtimeFPSTracker(window_size=25)

    def run(self):
        self.log_message.emit(f"Start Pipeline: source='{self.source}', device='{self.device}'")
        frame_queue = Queue(maxsize=30)
        detect_queue = Queue(maxsize=30)
        output_queue = Queue(maxsize=30)
        p_reader = Process(
            target=video_reader_process,
            args=(self.source, self.imgsz, 32, frame_queue, self.stop_event),
            name="UI_VideoReader"
        )
        p_detector = Process(
            target=detector_tracker_process,
            kwargs=dict(
                weights=self.weights,
                imgsz=self.imgsz,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                device=self.device,
                frame_queue=frame_queue,
                detect_queue=detect_queue,
                stop_event=self.stop_event
            ),
            name="UI_DetectorTracker"
        )
        p_decision = Process(
            target=feature_mlp_rule_process,
            args=(detect_queue, output_queue, self.stop_event),
            name="UI_DecisionEngine"
        )

        self.processes = [p_reader, p_detector, p_decision]
        for p in self.processes:
            p.daemon = True
            p.start()

        self.log_message.emit("Successfully!")
        save_path = None
        if self.save_video:
            out_dir = PROJECT_ROOT / "runs" / "ui_recordings"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = str(out_dir / f"adas_run_{timestamp}.mp4")
            self.log_message.emit(f"Video đầu ra sẽ được lưu tại: {save_path}")

        while self._is_running and not self.stop_event.is_set():
            if self._is_paused:
                time.sleep(0.05)
                continue
            try:
                data = output_queue.get(timeout=0.2)
            except Exception:
                if not p_reader.is_alive() and not p_decision.is_alive():
                    self.log_message.emit("Đã đọc hết video.")
                    break
                continue
            if data is None:
                break
            self.fps_tracker.update()
            real_fps = self.fps_tracker.get_fps()
            if len(data) >= 6:
                path, im0, s, current_speed, safe_distance, telemetry = data[:6]
            else:
                path, im0, s, current_speed, safe_distance = data[:5]
                telemetry = {
                    "current_speed": current_speed,
                    "safe_distance": safe_distance,
                    "objects": [],
                    "warning_level": "SAFE",
                    "warning_text": "An toàn - Không có nguy cơ"
                }

            speed_val = telemetry.get("current_speed", telemetry.get("speed_kmh", current_speed))
            dist_val = telemetry.get("safe_distance", telemetry.get("safe_dist_m", safe_distance))

            t_capture = telemetry.get("t_capture", None)
            if t_capture:
                latency_e2e = round((time.time() - t_capture) * 1000, 1)
            else:
                latency_e2e = telemetry.get("latency_ms", 0.0)

            telemetry["current_speed"] = int(speed_val)
            telemetry["safe_distance"] = float(dist_val)
            telemetry["real_fps"] = round(real_fps, 1)
            telemetry["latency_ms"] = latency_e2e
            if self.save_video and im0 is not None:
                if self.vid_writer is None:
                    h, w = im0.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.vid_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))
                self.vid_writer.write(im0)
            if im0 is not None:
                rgb_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                self.frame_ready.emit(q_img, telemetry)
        self.stop()
        if self.vid_writer is not None:
            self.vid_writer.release()
            self.vid_writer = None
            self.log_message.emit("Đã hoàn tất lưu file video.")

        self.pipeline_finished.emit()

    def pause(self):
        self._is_paused = not self._is_paused
        status = "TẠM DỪNG" if self._is_paused else "TIẾP TỤC"
        self.log_message.emit(f"⏸ Trạng thái pipeline: {status}")

    def stop(self):
        self._is_running = False
        self.stop_event.set()
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
        self.processes.clear()

class ADASDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADAS INTELLIGENT COLLISION WARNING SYSTEM")
        self.setMinimumSize(1280, 820)
        self.resize(1440, 920)
        self.worker = None
        sample_dir = PROJECT_ROOT / "runs" / "mlp_dataset_videos"
        sample_videos = sorted(list(sample_dir.glob("*.mp4"))) if sample_dir.exists() else []
        if (PROJECT_ROOT / "NO20250603-083005-009846F.MP4").exists():
            self.default_video = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
        elif sample_videos:
            self.default_video = sample_videos[0]
        else:
            self.default_video = PROJECT_ROOT / "NO20250603-083005-009846F.MP4"
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 12, 15, 15)
        main_layout.setSpacing(12)
        top_panel = self.create_top_panel()
        main_layout.addWidget(top_panel)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        left_container = self.create_video_container()
        splitter.addWidget(left_container)
        right_container = self.create_telemetry_container()
        splitter.addWidget(right_container)
        splitter.setSizes([950, 450])
        main_layout.addWidget(splitter, 1)

    def create_top_panel(self):
        panel = QGroupBox("⚙️ Cấu Hình Đầu Vào & Điều Khiển Hệ Thống")
        panel.setObjectName("ControlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(10)
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        lbl_source = QLabel("Nguồn:")
        lbl_source.setStyleSheet("font-weight: bold; color: #58a6ff;")
        self.cb_source_type = QComboBox()
        self.cb_source_type.addItems(["Video File", "Webcam / Camera", "RTSP / IP Camera"])
        self.cb_source_type.currentIndexChanged.connect(self.on_source_type_changed)

        self.txt_source = QLineEdit()
        if self.default_video.exists():
            self.txt_source.setText(str(self.default_video))
        else:
            self.txt_source.setText("0")

        self.btn_browse = QPushButton("📁 Chọn Video")
        self.btn_browse.clicked.connect(self.browse_video)

        lbl_device = QLabel("Thiết Bị:")
        lbl_device.setStyleSheet("font-weight: bold; color: #58a6ff;")
        self.cb_device = QComboBox()
        self.cb_device.addItem("CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.cb_device.addItem(f"CUDA:{i} ({torch.cuda.get_device_name(i)})")
            self.cb_device.setCurrentIndex(1)

        self.chk_save = QCheckBox("Lưu video MP4")
        self.chk_save.setChecked(False)
        row1.addWidget(lbl_source)
        row1.addWidget(self.cb_source_type)
        row1.addWidget(self.txt_source, 1)
        row1.addWidget(self.btn_browse)
        row1.addWidget(lbl_device)
        row1.addWidget(self.cb_device)
        row1.addWidget(self.chk_save)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        self.btn_start = QPushButton("▶ BẮT ĐẦU")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.clicked.connect(self.start_pipeline)
        self.btn_pause = QPushButton("⏸ TẠM DỪNG")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.setMinimumWidth(110)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_pipeline)
        self.btn_stop = QPushButton("⏹ DỪNG")
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.setMinimumWidth(100)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_pipeline)
        row2.addWidget(self.btn_start)
        row2.addWidget(self.btn_pause)
        row2.addWidget(self.btn_stop)
        v_sep = QFrame()
        v_sep.setFrameShape(QFrame.VLine)
        v_sep.setFrameShadow(QFrame.Sunken)
        v_sep.setStyleSheet("color: #30363d;")
        row2.addWidget(v_sep)
        self.badge_speed = QLabel("Vận tốc OCR: 0 km/h")
        self.badge_speed.setObjectName("BadgeSpeed")
        self.badge_speed.setStyleSheet("""
            QLabel#BadgeSpeed {
                background-color: #21262d;
                color: #f0883e;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #f0883e;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)

        self.badge_safe_dist = QLabel("Cự ly an toàn: 0.0 m")
        self.badge_safe_dist.setObjectName("BadgeSafeDist")
        self.badge_safe_dist.setStyleSheet("""
            QLabel#BadgeSafeDist {
                background-color: #21262d;
                color: #d29922;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #d29922;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)

        self.badge_fps = QLabel("⚡ Realtime FPS: 0.0")
        self.badge_fps.setObjectName("BadgeFPS")
        self.badge_fps.setStyleSheet("""
            QLabel#BadgeFPS {
                background-color: #21262d;
                color: #58a6ff;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #58a6ff;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)
        self.badge_latency = QLabel("⏱️ Độ trễ: 0.0 ms")
        self.badge_latency.setObjectName("BadgeLatency")
        self.badge_latency.setStyleSheet("""
            QLabel#BadgeLatency {
                background-color: #21262d;
                color: #3fb950;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #3fb950;
                border-radius: 6px;
                padding: 6px 14px;
            }
        """)
        row2.addWidget(self.badge_speed)
        row2.addWidget(self.badge_safe_dist)
        row2.addWidget(self.badge_fps)
        row2.addWidget(self.badge_latency)
        row2.addStretch(1)
        layout.addLayout(row2)
        return panel

    def create_video_container(self):
        container = QFrame()
        container.setObjectName("VideoContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.lbl_warning_banner = QLabel("HỆ THỐNG SẴN SÀNG")
        self.lbl_warning_banner.setObjectName("WarningBannerSafe")
        self.lbl_warning_banner.setAlignment(Qt.AlignCenter)
        self.lbl_warning_banner.setFixedHeight(45)
        layout.addWidget(self.lbl_warning_banner)
        self.lbl_video = QLabel("CHƯA CÓ TÍN HIỆU HÌNH ẢNH\nNhấn 'BẮT ĐẦU' để chạy kiểm thử video")
        self.lbl_video.setObjectName("VideoViewport")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(640, 480)
        self.lbl_video.setStyleSheet("background-color: #0d1117; color: #8b949e; border-radius: 8px; font-size: 16px;")
        layout.addWidget(self.lbl_video, 1)
        return container

    def create_telemetry_container(self):
        container = QFrame()
        container.setObjectName("TelemetryContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.card_fps = self.create_stat_card("FPS MÁY TÍNH (THỰC TẾ)", "--", "#58a6ff")
        self.card_latency = self.create_stat_card("ĐỘ TRỄ END-TO-END", "-- ms", "#3fb950")
        self.card_speed = self.create_stat_card("VẬN TỐC XE (OCR)", "-- km/h", "#f0883e")
        self.card_safe_dist = self.create_stat_card("KHOẢNG CÁCH AN TOÀN", "-- m", "#d29922")
        grid.addWidget(self.card_fps["frame"], 0, 0)
        grid.addWidget(self.card_latency["frame"], 0, 1)
        grid.addWidget(self.card_speed["frame"], 1, 0)
        grid.addWidget(self.card_safe_dist["frame"], 1, 1)
        layout.addLayout(grid)
        lbl_tbl = QLabel("Live Telemetry:")
        lbl_tbl.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(lbl_tbl)
        self.tbl_objects = QTableWidget()
        self.tbl_objects.setColumnCount(7)
        self.tbl_objects.setHorizontalHeaderLabels(["ID", "Class", "Cự Ly (m)", "TTC (s)", "Hazard (H)", "Dự Báo", "Góc (°)"])
        self.tbl_objects.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_objects.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_objects.setAlternatingRowColors(True)
        self.tbl_objects.verticalHeader().setVisible(False)
        self.tbl_objects.setFixedHeight(220)
        layout.addWidget(self.tbl_objects)
        lbl_log = QLabel("📋 Nhật Ký Hệ Thống:")
        lbl_log.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(lbl_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setObjectName("LogConsole")
        self.txt_log.setFixedHeight(140)
        layout.addWidget(self.txt_log)

        return container

    def create_stat_card(self, title, default_val, accent_color):
        frame = QFrame()
        frame.setObjectName("StatCard")
        frame.setStyleSheet(f"""
            QFrame#StatCard {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-left: 4px solid {accent_color};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold;")
        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet(f"color: {accent_color}; font-size: 20px; font-weight: bold;")
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        return {"frame": frame, "value": lbl_val}

    def on_source_type_changed(self, index):
        if index == 0:  # Video File
            self.txt_source.setText(str(self.default_video) if self.default_video.exists() else "")
            self.btn_browse.setEnabled(True)
        elif index == 1:  # Webcam
            self.txt_source.setText("0")
            self.btn_browse.setEnabled(False)
        elif index == 2:  # RTSP
            self.txt_source.setText("rtsp://admin:password@192.168.1.100:554/stream1")
            self.btn_browse.setEnabled(False)

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video Dashcam / Kiểm Thử",
            str(PROJECT_ROOT),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.txt_source.setText(file_path)

    def start_pipeline(self):
        source = self.txt_source.text().strip()
        if not source:
            self.append_log("❌ Lỗi: Chưa cung cấp nguồn video hoặc webcam.")
            return

        device_raw = self.cb_device.currentText()
        if "CUDA" in device_raw:
            device = device_raw.split(":")[1].split()[0]
        else:
            device = 'cpu'

        save_video = self.chk_save.isChecked()
        weights = [str(MODEL_DETECTION_PATH)]

        self.worker = PipelineWorker(
            source=source,
            weights=weights,
            imgsz=(320, 320),
            conf_thres=0.25,
            iou_thres=0.45,
            device=device,
            save_video=save_video
        )
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_message.connect(self.append_log)
        self.worker.pipeline_finished.connect(self.on_pipeline_finished)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.cb_source_type.setEnabled(False)
        self.txt_source.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.worker.start()

    def pause_pipeline(self):
        if self.worker:
            self.worker.pause()
            if self.worker._is_paused:
                self.btn_pause.setText("▶ TIẾP TỤC")
            else:
                self.btn_pause.setText("⏸ TẠM DỪNG")

    def stop_pipeline(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.on_pipeline_finished()
        self.append_log("🛑 Người dùng đã dừng hệ thống.")

    def on_pipeline_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ TẠM DỪNG")
        self.btn_stop.setEnabled(False)
        self.cb_source_type.setEnabled(True)
        self.txt_source.setEnabled(True)
        self.btn_browse.setEnabled(True)

    def update_frame(self, q_img, telemetry):
        # 1. Scale hình ảnh giữ tỉ lệ vừa vặn với Label
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.lbl_video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_video.setPixmap(scaled_pixmap)
        speed = telemetry.get("current_speed", telemetry.get("speed_kmh", 0))
        safe_dist = float(telemetry.get("safe_distance", telemetry.get("safe_dist_m", 0.0)))
        real_fps = telemetry.get("real_fps", telemetry.get("fps", 0.0))
        latency = telemetry.get("latency_ms", 0.0)
        self.badge_speed.setText(f"Vận tốc OCR: {speed} km/h")
        self.badge_safe_dist.setText(f"Cự ly an toàn: {safe_dist:.1f} m")
        self.badge_fps.setText(f"Realtime FPS: {real_fps:.1f}")
        self.badge_latency.setText(f"⏱️ Độ trễ: {latency:.1f} ms")
        self.card_fps["value"].setText(f"{real_fps:.1f}")
        self.card_latency["value"].setText(f"{latency:.1f} ms")
        self.card_speed["value"].setText(f"{speed} km/h")
        self.card_safe_dist["value"].setText(f"{safe_dist:.1f} m")
        warning_level = telemetry.get("warning_level", "SAFE")
        warning_text = telemetry.get("warning_text", "AN TOÀN")

        if warning_level == "CRITICAL":
            self.lbl_warning_banner.setStyleSheet("""
                background-color: #da3633;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            """)
            self.lbl_warning_banner.setText(f"🚨 CẢNH BÁO NGUY CƠ: {warning_text.upper()}!")
        elif warning_level == "WARNING":
            self.lbl_warning_banner.setStyleSheet("""
                background-color: #d29922;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            """)
            self.lbl_warning_banner.setText(f"⚠️ CHÚ Ý: {warning_text.upper()}")
        else:
            self.lbl_warning_banner.setStyleSheet("""
                background-color: #238636;
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            """)
            self.lbl_warning_banner.setText(f"🛡️ {warning_text.upper()}")
        objects = telemetry.get("objects", [])
        self.tbl_objects.setRowCount(len(objects))
        for row, obj in enumerate(objects):
            # ID
            item_id = QTableWidgetItem(f"#{obj.get('id', '-')}")
            item_id.setTextAlignment(Qt.AlignCenter)
            self.tbl_objects.setItem(row, 0, item_id)

            # Class
            item_cls = QTableWidgetItem(str(obj.get('class', '-')))
            item_cls.setTextAlignment(Qt.AlignCenter)
            self.tbl_objects.setItem(row, 1, item_cls)

            # Distance
            dist = obj.get('distance', 0.0)
            item_dist = QTableWidgetItem(f"{dist:.1f}")
            item_dist.setTextAlignment(Qt.AlignCenter)
            self.tbl_objects.setItem(row, 2, item_dist)

            # TTC (Time-To-Collision)
            ttc_val = obj.get('ttc', 99.9)
            ttc_str = f"{ttc_val:.1f}s" if ttc_val < 90.0 else "∞"
            item_ttc = QTableWidgetItem(ttc_str)
            item_ttc.setTextAlignment(Qt.AlignCenter)
            if ttc_val < 3.0:
                item_ttc.setForeground(QColor("#f85149"))
                item_ttc.setFont(QFont("Arial", 9, QFont.Bold))
            self.tbl_objects.setItem(row, 3, item_ttc)

            # Hazard Score H_i(t)
            h_val = obj.get('hazard_score', 0.0)
            item_h = QTableWidgetItem(f"{h_val:.2f}")
            item_h.setTextAlignment(Qt.AlignCenter)
            if h_val >= 0.5:
                item_h.setForeground(QColor("#f85149"))
                item_h.setFont(QFont("Arial", 9, QFont.Bold))
            elif h_val >= 0.3:
                item_h.setForeground(QColor("#e3b341"))
            self.tbl_objects.setItem(row, 4, item_h)

            # Status Behavior (Event Classification & Warning Decision)
            status = obj.get('status', 'normal')
            item_status = QTableWidgetItem(status)
            item_status.setTextAlignment(Qt.AlignCenter)

            if status == "wrong-way":
                item_status.setForeground(QColor("#f85149"))
                item_status.setFont(QFont("Arial", 9, QFont.Bold))
            elif status == "crossing":
                item_status.setForeground(QColor("#e3b341"))
                item_status.setFont(QFont("Arial", 9, QFont.Bold))
            else:
                item_status.setForeground(QColor("#3fb950"))
            self.tbl_objects.setItem(row, 5, item_status)

            # Angle
            angle = obj.get('angle', 0.0)
            item_angle = QTableWidgetItem(f"{angle:.1f}°")
            item_angle.setTextAlignment(Qt.AlignCenter)
            self.tbl_objects.setItem(row, 6, item_angle)

    def append_log(self, msg):
        timestamp = time.strftime("[%H:%M:%S]")
        self.txt_log.append(f"{timestamp} {msg}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

    def apply_dark_theme(self):
        """
        Thiết lập giao diện Dark Theme phong cách buồng lái xe thông minh (ADAS HUD Cockpit).
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0e14;
            }
            QWidget {
                color: #c9d1d9;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 14px;
                font-weight: bold;
                color: #58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
            }
            QLineEdit, QComboBox {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #58a6ff;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: bold;
                color: #c9d1d9;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton#BtnStart {
                background-color: #238636;
                border-color: #2ea043;
                color: #ffffff;
            }
            QPushButton#BtnStart:hover {
                background-color: #2ea043;
            }
            QPushButton#BtnPause {
                background-color: #9e6a03;
                border-color: #bb8009;
                color: #ffffff;
            }
            QPushButton#BtnPause:hover {
                background-color: #bb8009;
            }
            QPushButton#BtnStop {
                background-color: #da3633;
                border-color: #f85149;
                color: #ffffff;
            }
            QPushButton#BtnStop:hover {
                background-color: #f85149;
            }
            QTableWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                gridline-color: #21262d;
            }
            QHeaderView::section {
                background-color: #0d1117;
                color: #8b949e;
                font-weight: bold;
                border: 1px solid #21262d;
                padding: 5px;
            }
            QTextEdit#LogConsole {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #7ee787;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)


def launch_dashboard():
    app = QApplication(sys.argv)
    window = ADASDashboard()
    window.show()
    sys.exit(app.exec_())
