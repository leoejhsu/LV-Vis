from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import psutil, os
from collections import deque
from PyQt5.QtWidgets import QComboBox, QLineEdit, QPushButton, QGroupBox
from PyQt5.QtWidgets import QPushButton, QFileDialog
import imageio
from datetime import datetime
import csv
import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception as e:
    print("⚠️ NVIDIA NVML not available:", e)
    HAS_NVML = False


class ControlPanel(QWidget):
    def __init__(self, volume_data, parent=None, history_length=60, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.history_length = history_length

        # time series ：GPU and CPU memory (MB)
        self.gpu_history = deque([0] * history_length, maxlen=history_length)
        self.mem_history = deque([0] * history_length, maxlen=history_length)

        # # Process handles
        # self.process = psutil.Process(os.getpid())
        # self._build_ui()
        # self._start_monitor()

        # Logging state
        self._logging = False
        self._log_rows = []           # list of (iso, elapsed_s, ram_mb, vram_mb)
        self._log_t0 = None           # datetime

        # Process handles
        self.process = psutil.Process(os.getpid())
        self._build_ui()
        self._start_monitor()

    def _build_ui(self):

        main_lay = QHBoxLayout()
        self.setLayout(main_lay)

        # Left: GPU process memory
        left = QWidget()
        llay = QVBoxLayout(left)
        lbl_g_title = QLabel("GPU Process Memory Over Time")
        lbl_g_title.setStyleSheet("font: bold 24px; color: black;")

        llay.addWidget(lbl_g_title)
        self.gpu_label = QLabel("GPU: 0.0 MB")
        self.gpu_label.setStyleSheet("font: bold 24px; color: black;")
        llay.addWidget(self.gpu_label)

        gpu_box = QWidget()
        gb = QHBoxLayout(gpu_box)
        # GPU time-series plot
        self.gpu_fig = plt.Figure(figsize=(6, 3), facecolor='black')
        self.gpu_ax = self.gpu_fig.add_subplot(111, facecolor='black')
        self.gpu_line, = self.gpu_ax.plot([], [], lw=1.5,
                                          marker='o', markersize=6,
                                          color='cyan')
        self.gpu_ax.set_ylim(0, 10)
        self.gpu_ax.set_xlim(0, self.history_length - 1)
        self.gpu_ax.set_xticks([])
        self.gpu_ax.tick_params(axis='both', labelsize=18, colors='cyan')
        self.gpu_ax.set_ylabel("GPU Mem (MB)", fontsize=18, color='cyan')

        self.gpu_canvas = FigureCanvas(self.gpu_fig)
        self.gpu_canvas.setMinimumHeight(300)
        self.gpu_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gb.addWidget(self.gpu_canvas, 1)

        llay.addWidget(gpu_box)
        main_lay.addWidget(left, 1)

        # Right: Python process memory
        right = QWidget()
        rlay = QVBoxLayout(right)
        lbl_m_title = QLabel("Python Process Memory Over Time")
        lbl_m_title.setStyleSheet("font: bold 24px; color: black;")
        rlay.addWidget(lbl_m_title)
        self.mem_label = QLabel("Memory: 0.0 MB")
        self.mem_label.setStyleSheet("font: bold 24px; color: black;")
        rlay.addWidget(self.mem_label)

        mem_box = QWidget()
        mb = QHBoxLayout(mem_box)

        self.mem_fig = plt.Figure(figsize=(6, 3), facecolor='black')
        self.mem_ax = self.mem_fig.add_subplot(111, facecolor='black')
        self.mem_line, = self.mem_ax.plot([], [], lw=1.5,
                                          marker='o', markersize=6,
                                          color='white')
        self.mem_ax.set_ylim(0, 10)
        self.mem_ax.set_xlim(0, self.history_length - 1)
        self.mem_ax.set_xticks([])
        self.mem_ax.tick_params(axis='both', labelsize=18, colors='white')
        self.mem_ax.set_ylabel("RAM (MB)", fontsize=18, color='white')

        self.mem_canvas = FigureCanvas(self.mem_fig)
        self.mem_canvas.setMinimumHeight(300)
        self.mem_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mb.addWidget(self.mem_canvas, 1)

        rlay.addWidget(mem_box)
        main_lay.addWidget(right, 1)

        # Camera Control
        camera_group = QGroupBox("Camera Control")
        camera_layout = QVBoxLayout()

        self.canvas_selector = QComboBox()
        self.canvas_selector.addItems(["Interactive", "Global"])

        self.axis_selector = QComboBox()
        self.axis_selector.addItems(["azimuth", "elevation", "roll"])

        self.input_value = QLineEdit()
        self.input_value.setPlaceholderText("Enter angle")

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_camera_rotation)

        screenshot_btn = QPushButton("Screenshot Interactive Canvas")
        screenshot_btn.setStyleSheet("font: bold 18px; color: black; background-color: lightgray;")
        screenshot_btn.clicked.connect(self._screenshot_canvas)

        # Add layout
        camera_layout.addWidget(self.canvas_selector)
        camera_layout.addWidget(self.axis_selector)
        camera_layout.addWidget(self.input_value)
        camera_layout.addWidget(self.apply_button)
        camera_layout.addWidget(screenshot_btn)
        center_button = QPushButton("Center Camera")
        center_button.setStyleSheet("font: bold 18px; color: black; background-color: lightgray;")
        center_button.clicked.connect(self._center_camera)
        camera_layout.addWidget(center_button)
        camera_group.setLayout(camera_layout)

        main_lay.addWidget(camera_group)

        # ====== Volume Panel ======
        vol_group = QGroupBox("Volume")
        vol_lay = QVBoxLayout()

        # 1) Drop-down: show by loader.id
        from PyQt5.QtWidgets import QDoubleSpinBox, QFormLayout
        self.volume_selector = QComboBox()
        self.volume_selector.currentIndexChanged.connect(self._on_volume_changed)
        vol_lay.addWidget(QLabel("Select Volume (by id)"))
        vol_lay.addWidget(self.volume_selector)

        # 2) Show transforms (interactive / global)
        self.lbl_tr_interactive = QLabel("Interactive translate: (—, —, —)")
        self.lbl_tr_global = QLabel("Global translate: (—, —, —)")
        vol_lay.addWidget(self.lbl_tr_interactive)
        vol_lay.addWidget(self.lbl_tr_global)

        # 3) Rendering params: gamma / u_min / u_max
        form = QFormLayout()
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.05, 5.0)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.valueChanged.connect(self._apply_render_params)

        self.spin_umin = QDoubleSpinBox()
        self.spin_umin.setRange(0.0, 1.0)
        self.spin_umin.setSingleStep(0.01)
        self.spin_umin.setValue(0.0)
        self.spin_umin.valueChanged.connect(self._apply_render_params)

        self.spin_umax = QDoubleSpinBox()
        self.spin_umax.setRange(0.0, 1.0)
        self.spin_umax.setSingleStep(0.01)
        self.spin_umax.setValue(1.0)
        self.spin_umax.valueChanged.connect(self._apply_render_params)

        form.addRow("gamma", self.spin_gamma)
        form.addRow("u_min", self.spin_umin)
        form.addRow("u_max", self.spin_umax)
        vol_lay.addLayout(form)

        # 4) Reset button
        self.btn_reset_render = QPushButton("Reset render params")
        self.btn_reset_render.clicked.connect(self._reset_render_params)
        vol_lay.addWidget(self.btn_reset_render)

        vol_group.setLayout(vol_lay)
        main_lay.addWidget(vol_group)

        # ====== Logging group ======
        log_group = QGroupBox("Memory Logging")
        log_lay = QVBoxLayout()

        # Status indicator: red REC, gray idle
        self.log_status = QLabel("● idle")
        self.log_status.setStyleSheet("font: bold 18px; color: gray;")
        self.log_toggle_btn = QPushButton("Start Logging")
        self.log_toggle_btn.setStyleSheet("font: bold 18px; color: black; background-color: lightgray;")
        self.log_toggle_btn.clicked.connect(self._toggle_logging)

        log_lay.addWidget(self.log_status)
        log_lay.addWidget(self.log_toggle_btn)
        log_group.setLayout(log_lay)
        main_lay.addWidget(log_group)

    def _start_monitor(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_usage)
        self._timer.start(1000)


    def _apply_camera_rotation(self):

        # offset = np.array([80., 80., 67.5])
        canvas_choice = self.canvas_selector.currentText()
        cam = (self.main_window.camera_controller.camera
               if canvas_choice == "Interactive"
               else self.main_window.global_camera_controller.camera)
        # cam.center += offset
        print("📍 Temp Center =", cam.center)


        axis = self.axis_selector.currentText()
        try:
            value = float(self.input_value.text())
            setattr(cam, axis, value)
        except ValueError:
            print("Invalid input")


        self.main_window.canvas_manager.canvas.update()
        self.main_window.global_canvas_manager.canvas.update()

    def _screenshot_canvas(self):
        if not hasattr(self.main_window, "canvas_manager"):
            print("Can't find canvas_manager")
            return

        # Temporarily hide the axis
        self.main_window.axis_vis.axis.visible = False

        canvas = self.main_window.canvas_manager.canvas
        img = canvas.render(alpha=True)[..., :3]

        img_cropped = crop_black_border(img)
        # from skimage.transform import resize
        # img_cropped = resize(img_cropped, (200, 200), preserve_range=True).astype(img_cropped.dtype)

        # Restore the axis
        self.main_window.axis_vis.axis.visible = True

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "screenshot.tiff", "TIFF Files (*.tiff)")
        if save_path:
            imageio.imwrite(save_path, img_cropped)
            print(f"Saved cropped screenshot to {save_path}")

    def _center_camera(self):
        if hasattr(self.main_window, "_recenter_camera_on_volumes"):
            self.main_window._recenter_camera_on_volumes()
            print("📌 Camera recentered to volumes.")
        else:
            print("Can't find _recenter_camera_on_volumes method")

    def _update_usage(self):
        # -- GPU process memory
        if HAS_NVML:
            try:
                try:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(_gpu_handle)
                except AttributeError:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v2(_gpu_handle)

                gpu_mb = 0.0
                for p in procs:
                    if p.pid == os.getpid():
                        gpu_mb = p.usedGpuMemory / 1024 ** 2
                        break
            except Exception as e:
                print("⚠️ Failed to query GPU usage:", e)
                gpu_mb = 0.0
        else:
            gpu_mb = 0.0

        self.gpu_history.append(gpu_mb)
        max_g = max(self.gpu_history)
        self.gpu_ax.set_ylim(0, max(max_g * 1.2, 10))
        x = list(range(len(self.gpu_history)))
        self.gpu_line.set_data(x, list(self.gpu_history))
        self.gpu_label.setText(f"GPU: {gpu_mb:.1f} MB")
        self.gpu_ax.set_xlim(0, self.history_length - 1)
        self.gpu_canvas.draw_idle()

        # -- Python process memory
        mem_mb = self.process.memory_info().rss / 1024 ** 2
        self.mem_history.append(mem_mb)
        max_m = max(self.mem_history)
        self.mem_ax.set_ylim(0, max(max_m * 1.2, 10))
        x = list(range(len(self.mem_history)))
        self.mem_line.set_data(x, list(self.mem_history))
        self.mem_label.setText(f"Memory: {mem_mb:.1f} MB")
        self.mem_ax.set_xlim(0, self.history_length - 1)
        self.mem_canvas.draw_idle()

        if self._logging:
            now = datetime.now()
            elapsed = (now - self._log_t0).total_seconds()
            self._log_rows.append(
                [now.isoformat(timespec='seconds'), f"{elapsed:.0f}", f"{mem_mb:.2f}", f"{gpu_mb:.2f}"])


    def _toggle_logging(self):
        if not self._logging:
            # Start
            self._logging = True
            self._log_rows.clear()
            self._log_t0 = datetime.now()
            self.log_status.setText("● REC")
            self.log_status.setStyleSheet("font: bold 18px; color: red;")
            self.log_toggle_btn.setText("Stop & Save")
            print("▶️ Memory logging started.")
        else:
            self._logging = False
            self.log_status.setText("● idle")
            self.log_status.setStyleSheet("font: bold 18px; color: gray;")
            self.log_toggle_btn.setText("Start Logging")
            print("⏹ Memory logging stopped.")
            self._save_log_to_csv()

    def _save_log_to_csv(self):
        if not self._log_rows:
            print("No data recorded. Skipping save.")
            return

        ts = self._log_t0.strftime("%Y%m%d_%H%M%S")
        default_name = f"memlog_{ts}.csv"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Log", default_name, "CSV Files (*.csv)")
        if not save_path:
            print("Save canceled.")
            return

        # Ensure .csv extension if user omitted it
        if not os.path.splitext(save_path)[1]:
            save_path += ".csv"

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_iso", "elapsed_s", "ram_mb", "vram_mb"])
                writer.writerows(self._log_rows)
            print(f"Saved log: {save_path} ({len(self._log_rows)} rows).")
        except Exception as e:
            print(f"Failed to save log: {e}")

    def populate_volumes(self, volumes_dict: dict):
        self._volumes = volumes_dict
        self.volume_selector.blockSignals(True)
        self.volume_selector.clear()
        ids = sorted(list(self._volumes.keys()))
        for vid in ids:
            self.volume_selector.addItem(str(vid), userData=vid)
        self.volume_selector.blockSignals(False)
        if ids:
            self.volume_selector.setCurrentIndex(0)
            self._on_volume_changed(0)

    def current_loader(self):
        if not hasattr(self, "_volumes") or self._volumes is None or self.volume_selector.count() == 0:
            return None
        vid = self.volume_selector.currentData()
        return self._volumes.get(vid, None)

    def _on_volume_changed(self, _idx):
        ldr = self.current_loader()
        if ldr is None:
            self.lbl_tr_interactive.setText("Interactive translate: (—, —, —)")
            self.lbl_tr_global.setText("Global translate: (—, —, —)")
            return
        lvl = self._current_level()
        self._update_transform_labels(ldr, lvl)
        self._read_render_params(ldr)

    def _update_transform_labels(self, ldr, lvl):
        f = ldr.factor[lvl] if hasattr(ldr, "factor") and len(ldr.factor) > 0 else 1
        f0 = ldr.factor[0] if hasattr(ldr, "factor") and len(ldr.factor) > 0 else 1

        if ldr.volume_visuals is not None and ldr.volume_visuals.transform is not None:
            t = ldr.volume_visuals.transform.translate
            txt = self._format_translate(t, f)
            self.lbl_tr_interactive.setText(f"Interactive translate: {txt}")
        else:
            self.lbl_tr_interactive.setText("Interactive translate: x=—  y=—  z=—")

        if ldr.gloVolume_visuals is not None and ldr.gloVolume_visuals.transform is not None:
            tg = ldr.gloVolume_visuals.transform.translate
            txtg = self._format_translate(tg, f0)
            self.lbl_tr_global.setText(f"Global translate: {txtg}")
        else:
            self.lbl_tr_global.setText("Global translate: x=—  y=—  z=—")

    def _read_render_params(self, ldr):
        vis = ldr.volume_visuals
        if vis is None:
            return
        try:
            gamma = getattr(vis, "gamma", 1.0)
        except Exception:
            gamma = 1.0
        self.spin_gamma.blockSignals(True)
        self.spin_gamma.setValue(float(gamma))
        self.spin_gamma.blockSignals(False)

        try:
            umin = float(vis.shared_program['u_min_val'])
            umax = float(vis.shared_program['u_max_val'])
        except Exception:
            umin, umax = 0.0, 1.0

        self.spin_umin.blockSignals(True)
        self.spin_umax.blockSignals(True)
        self.spin_umin.setValue(umin)
        self.spin_umax.setValue(umax)
        self.spin_umin.blockSignals(False)
        self.spin_umax.blockSignals(False)

    def _apply_render_params(self, *_):
        ldr = self.current_loader()
        if ldr is None:
            return
        gamma = float(self.spin_gamma.value())
        umin = float(self.spin_umin.value())
        umax = float(self.spin_umax.value())
        if umax < umin:
            umax = umin
            self.spin_umax.blockSignals(True)
            self.spin_umax.setValue(umax)
            self.spin_umax.blockSignals(False)

        for vis in (ldr.volume_visuals, ldr.gloVolume_visuals):
            if vis is None:
                continue
            try:
                vis.gamma = gamma
            except Exception:
                pass
            try:
                vis.shared_program['u_min_val'] = umin
                vis.shared_program['u_max_val'] = umax
            except Exception:
                pass

        self.main_window.canvas_manager.canvas.update()
        self.main_window.global_canvas_manager.canvas.update()
        self._update_transform_labels(ldr, self._current_level())

    def _reset_render_params(self):
        self.spin_gamma.setValue(1.0)
        self.spin_umin.setValue(0.0)
        self.spin_umax.setValue(1.0)
        self._apply_render_params()

    def _format_translate(self, translate, factor0):
        vals = list(translate)
        vals = vals[:3]
        x = int(round(float(vals[0]) * float(factor0)))
        y = int(round(float(vals[1]) * float(factor0)))
        z = int(round(float(vals[2]) * float(factor0)))
        return f"x={x}  y={y}  z={z}"

    def refresh_current(self):
        ldr = self.current_loader()
        if ldr is None:
            return
        self._update_transform_labels(ldr, self._current_level())
        self._read_render_params(ldr)

    def _current_level(self) -> int:
        try:
            return int(self.main_window.controller.current_layer)
        except Exception:
            return 0



def crop_black_border(image: np.ndarray, threshold: int = 10, margin: int = 0) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    mask = gray > threshold
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(image.shape[0], y1 + margin)
    x1 = min(image.shape[1], x1 + margin)

    return image[y0:y1, x0:x1]
