
"""
rtsp_viewer_pyqt_safe_stop.py
Multi-stream RTSP viewer (PyQt5) with safer Stop behavior.
Each stream runs in its own QThread using a worker that uses capture.grab()/retrieve()
to minimize blocking and allows graceful stop without freezing the GUI.

Requirements:
    pip install opencv-python PyQt5

Usage:
    python rtsp_viewer_pyqt_safe_stop.py
Add streams in the UI, then press Start/Stop per stream.
"""
global count
count = 0
import sys
import os
import cv2
import time
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets


class StreamWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    global count
    count = 0
    def __init__(self, url, backend='ffmpeg', protocol='tcp', record=False, output_dir='.', parent=None):
        super().__init__(parent)
        self.url = url
        self.backend = backend
        self.protocol = protocol
        self.record = record
        self.output_dir = output_dir

        self._stop = False
        self.cap = None
        self.writer = None

    def build_url(self):
        url = self.url
        if self.backend == 'ffmpeg':
            # append protocol as simple query flag (FFmpeg interprets various ways)
            if '?' not in url:
                url = f"{url}?{self.protocol}"
            else:
                url = f"{url}&{self.protocol}"
        return url

    @QtCore.pyqtSlot()
    def run(self):
        backoff = 1.0
        while not self._stop:
            open_url = self.build_url()
            self.status.emit("Connecting...")
            try:
                if self.backend == 'ffmpeg':
                    self.cap = cv2.VideoCapture(open_url, cv2.CAP_FFMPEG)
                elif self.backend == 'gstreamer':
                    gst = (
                        f'rtspsrc location={self.url} latency=200 ! rtph264depay ! h264parse ! '
                        'avdec_h264 ! videoconvert ! appsink'
                    )
                    self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                else:
                    self.cap = cv2.VideoCapture(open_url)
            except Exception as e:
                self.status.emit(f"Open error: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            if not self.cap or not self.cap.isOpened():
                self.status.emit("Failed to open. Retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            # try to reduce internal buffering
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps != fps:
                fps = 20.0

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            if self.record:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c for c in self.url if c.isalnum() or c in "._-")[:40]
                out_path = os.path.join(self.output_dir, f"{safe_name}_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                try:
                    self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    self.status.emit(f"Recording -> {out_path}")
                except Exception as e:
                    self.status.emit(f"Writer error: {e}")
                    self.writer = None

            self.status.emit("Streaming")
            backoff = 1.0

            # Main loop: use grab/retrieve to avoid long blocking on read()
            while not self._stop:
                try:
                    grabbed = self.cap.grab()
                except Exception as e:
                    self.status.emit(f"Grab error: {e}")
                    break

                if not grabbed:
                    # failed to grab frame => reconnect
                    self.status.emit("Frame grab failed, reconnecting...")
                    break

                # retrieve the grabbed frame
                try:
                    ret, frame = self.cap.retrieve()
                except Exception as e:
                    # fallback to read if retrieve not available
                    try:
                        ret, frame = self.cap.read()
                    except Exception as ex2:
                        self.status.emit(f"Retrieve/read error: {ex2}")
                        break

                if not ret or frame is None:
                    self.status.emit("No frame, reconnecting...")
                    break

                # minimal corrupted-frame check
                try:
                    h, w = frame.shape[:2]
                except Exception:
                    self.status.emit("Corrupted frame, reconnecting...")
                    break

                # record if needed
                if self.writer is not None:
                    try:
                        self.writer.write(frame)
                    except Exception as e:
                        self.status.emit(f"Recording error: {e}")
                        try:
                            self.writer.release()
                        except:
                            pass
                        self.writer = None

                # emit frame to UI
                self.frame_ready.emit(frame)

                # small sleep to yield CPU; respect fps
                # avoid sleeping too long so stop() is responsive
                time.sleep(max(0.005, 1.0 / fps))

            # cleanup capture/writer on disconnect or stop
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
            if self.writer:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None

            if not self._stop:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        self.status.emit("Stopped")
        self.finished.emit()

    def stop(self):
        # Signal the loop to stop. The loop checks this frequently.
        self._stop = True
        # try to release capture immediately to unblock grab/read
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass


class StreamWidget(QtWidgets.QWidget):
    def __init__(self, url, backend='ffmpeg', protocol='tcp', record=False, parent=None):
        super().__init__(parent)
        self.url = url
        self.backend = backend
        self.protocol = protocol
        self.record = record

        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("No Video")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.layout.addWidget(self.label)

        self.lbl_status = QtWidgets.QLabel("Idle")
        self.layout.addWidget(self.lbl_status)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        self.layout.addLayout(btn_layout)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        self.thread = None
        self.worker = None

    def start(self):
        if self.thread and self.thread.isRunning():
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = StreamWorker(self.url, backend=self.backend, protocol=self.protocol, record=self.record)
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.finished.connect(self.on_finished)
        self.thread.start()

    def stop(self):
        self.btn_stop.setEnabled(False)
        if self.worker:
            self.worker.stop()
        # wait for thread to finish with timeout, but don't block UI thread
        QtCore.QTimer.singleShot(100, self._cleanup_thread)

    def _cleanup_thread(self):
        if self.thread:
            # ask thread to quit; worker will emit finished when done
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")
        self.label.setText("Stopped")

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f'frame{count}', frame_rgb)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.label.setPixmap(pix.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            ts1 = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'frame{ts1}.jpg', frame_rgb)
            print(f'done {ts1}')
            # count+=1
        except Exception as e:
            self.lbl_status.setText(f"Render error: {e}")

    @QtCore.pyqtSlot(str)
    def on_status(self, text):
        self.lbl_status.setText(text)

    @QtCore.pyqtSlot()
    def on_finished(self):
        # Called when worker reports finished
        self._cleanup_thread()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTSP Multi-Stream Viewer (Safe Stop)")
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.layout = QtWidgets.QVBoxLayout(self.central)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        self.input_url = QtWidgets.QLineEdit()
        self.input_url.setPlaceholderText("rtsp://user:pass@ip:port/stream")
        self.cbo_backend = QtWidgets.QComboBox()
        self.cbo_backend.addItems(["ffmpeg", "gstreamer", "default"])
        self.cbo_proto = QtWidgets.QComboBox()
        self.cbo_proto.addItems(["tcp", "udp"])
        self.chk_record = QtWidgets.QCheckBox("Record")
        self.btn_add = QtWidgets.QPushButton("Add Stream")

        ctrl.addWidget(QtWidgets.QLabel("URL:"))
        ctrl.addWidget(self.input_url)
        ctrl.addWidget(QtWidgets.QLabel("Backend:"))
        ctrl.addWidget(self.cbo_backend)
        ctrl.addWidget(QtWidgets.QLabel("Proto:"))
        ctrl.addWidget(self.cbo_proto)
        ctrl.addWidget(self.chk_record)
        ctrl.addWidget(self.btn_add)
        self.layout.addLayout(ctrl)

        # Scroll area for streams
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.stream_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.stream_container)
        self.scroll.setWidget(self.stream_container)
        self.layout.addWidget(self.scroll)

        self.stream_widgets = []
        self.btn_add.clicked.connect(self.add_stream)

        self.statusBar().showMessage("Ready")

    def add_stream(self):
        url = self.input_url.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Input", "Please enter RTSP URL")
            return
        backend = self.cbo_backend.currentText()
        proto = self.cbo_proto.currentText()
        record = self.chk_record.isChecked()

        widget = StreamWidget(url, backend=backend, protocol=proto, record=record)
        row = len(self.stream_widgets) // 2
        col = len(self.stream_widgets) % 2
        self.grid.addWidget(widget, row, col)
        self.stream_widgets.append(widget)
        widget.show()
        self.input_url.clear()

    def closeEvent(self, event):
        # Stop all workers gracefully
        for w in self.stream_widgets:
            w.stop()
        # allow a short time for threads to exit
        QtWidgets.QApplication.processEvents()
        time.sleep(0.1)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
