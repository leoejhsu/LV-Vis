# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh

"""
A simple PyQt5 GUI wrapper for your preprocessing pipeline
(data_preprocessing_adpated.py). Designed for non‑programmers.

Key features:
- Three input modes: CSV / Folder of TIFFs / Single 3D TIFF stack
- DType preset (uint16 / float32), min_size, output name
- Dry‑Run (estimate): reads metadata, computes max LOD level & padded shape
- Start / Cancel with threaded execution (responsive UI)
- Live log pane + rolling logfile next to input data
- Progress indicator (stage‑based) and error popups

How to run:
    python preprocess_gui.py

Assumptions:
- data_preprocessing_adpated.py is importable (same folder or PYTHONPATH).
- Your original functions remain unchanged; we call them as a black box.

Notes:
- If you later expose fine‑grained progress from crop/assemble via callbacks,
  you can wire them to Worker.report_progress to show % in the bar.
"""
import os
import sys
import traceback
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QComboBox, QSpinBox, QGroupBox, QProgressBar, QPlainTextEdit,
    QMessageBox, QCheckBox
)

# --- import your existing preprocessing code ---
try:
    import numpy as np
    from preprocess import (
        main as preprocess_main,
        oneVolProcessing,
        compute_maximum_level,
        LazyTiffSlices
    )
    import tifffile as tiff
except Exception as e:
    print("[FATAL] Could not import pipeline:", e)
    raise

os.environ["TQDM_DISABLE"] = "1"

# ------------------------ Worker Thread ------------------------ #
class Worker(QThread):
    started_human = pyqtSignal(str)
    log = pyqtSignal(str)
    stage = pyqtSignal(str)
    progress = pyqtSignal(int)      # 0-100 (coarse)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, dtype_name: str, *, dry_run: bool,
                 custom_min_size: int | None = None, lod_name: str = "LOD_Data", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.dtype_name = dtype_name
        self.dry_run = dry_run
        self.custom_min_size = custom_min_size
        self.lod_name = lod_name
        self._abort = False

    def request_abort(self):
        self._abort = True

    def run(self):
        try:
            self.started_human.emit("Start Execution")
            self.progress.emit(5)

            if self.dry_run:
                self._dry_run_estimate()
                self.progress.emit(100)
                self.finished_ok.emit()
                return

            self.stage.emit("Preparing input and output environment…")
            self.progress.emit(10)

            # dtype: Support Auto (use None to let backend decide based on image0.dtype)
            if self.dtype_name.startswith("Auto"):
                dtype = None
            else:
                dtype = getattr(np, self.dtype_name)

            # Infer mode (CSV / Folder / TIFF)
            mode = self._infer_mode(self.file_path)
            self.log.emit(f"[Info] Detected mode: {mode}")

            log_dir = self._default_output_dir(self.file_path)
            os.makedirs(log_dir, exist_ok=True)

            self.stage.emit("Launching preprocessing pipeline…")
            self.progress.emit(20)

            # irectly call oneVolProcessing, pass lod_name / callbacks
            if mode == "CSV":
                import pandas as pd
                df = pd.read_csv(self.file_path, header=None)
                total = len(df)
                for idx, row in df.iterrows():
                    if self._abort:
                        break
                    folder_or_stack = row[0]
                    if os.path.isdir(folder_or_stack):
                        oneVolProcessing(
                            folder_or_stack, dtype,
                            progress_cb=lambda p: self.progress.emit(max(20, min(99, p))),
                            stage_cb=lambda s: self.stage.emit(f"[{idx + 1}/{total}] {s}"),
                            abort_cb=lambda: self._abort,
                            lod_name=self.lod_name,
                            min_size=self.custom_min_size
                        )
                    else:
                        parent_dir = os.path.dirname(folder_or_stack)
                        oneVolProcessing(
                            parent_dir, dtype, stack_path=folder_or_stack,
                            progress_cb=lambda p: self.progress.emit(max(20, min(99, p))),
                            stage_cb=lambda s: self.stage.emit(f"[{idx + 1}/{total}] {s}"),
                            abort_cb=lambda: self._abort,
                            lod_name=self.lod_name,
                            min_size=self.custom_min_size
                        )
            elif mode == "FOLDER":
                oneVolProcessing(
                    self.file_path, dtype,
                    progress_cb=lambda p: self.progress.emit(max(20, min(99, p))),
                    stage_cb=lambda s: self.stage.emit(s),
                    abort_cb=lambda: self._abort,
                    lod_name=self.lod_name,
                    min_size=self.custom_min_size
                )
            elif mode == "TIFF":
                parent_dir = os.path.dirname(self.file_path)
                oneVolProcessing(
                    parent_dir, dtype, stack_path=self.file_path,
                    progress_cb=lambda p: self.progress.emit(max(20, min(99, p))),
                    stage_cb=lambda s: self.stage.emit(s),
                    abort_cb=lambda: self._abort,
                    lod_name=self.lod_name,
                    min_size=self.custom_min_size
                )
            else:
                raise ValueError("Unsupported input mode")

            self.progress.emit(100)
            self.stage.emit("Completed")
            self.finished_ok.emit()

        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")

    # --- helper: quick estimate without heavy work ---
    def _dry_run_estimate(self):
        path = self.file_path
        self.stage.emit("Reading image metadata (Dry-Run)…")
        self.progress.emit(25)

        try:
            mode = self._infer_mode(path)
            if mode == "TIFF":
                with tiff.TiffFile(path) as tif:
                    z = len(tif.pages)
                    y, x = tif.pages[0].asarray().shape
            elif mode == "FOLDER":
                import re
                items = [i for i in os.listdir(path) if i.lower().endswith(('.tif', '.tiff'))]
                items = sorted(items, key=lambda s: int(re.findall(r"\d+", s)[-1])) if items else items
                import tifffile as tiff_local
                arr0 = tiff_local.imread(os.path.join(path, items[0]))
                z, y, x = len(items), arr0.shape[0], arr0.shape[1]
            elif mode == "CSV":
                import pandas as pd
                df = pd.read_csv(path, header=None)
                samp = df.iloc[0, 0]
                return self._dry_run_estimate_path(samp)
            else:
                raise ValueError("Unknown mode")
        except Exception:
            # fallback: try as single tiff anyway
            with tiff.TiffFile(path) as tif:
                z = len(tif.pages)
                y, x = tif.pages[0].asarray().shape

        shape = (z, y, x)
        self.log.emit(f"Image shape: Z={z}, Y={y}, X={x}")

        min_size = self.custom_min_size if self.custom_min_size else 100
        max_level = self._compute_max_level(shape, min_size=min_size)
        self.log.emit(f"Suggested min_size={min_size} → Estimated LOD levels={max_level}")

        import numpy as np
        pad = np.ceil(np.array(shape) / (2 ** max_level)).astype(int)
        self.log.emit(f"Estimated padded unit shape: {tuple(pad)};\n"
                      f"The program will generate 8^{max_level} blocks accordingly")

    def _dry_run_estimate_path(self, folder_or_tiff):
        if os.path.isdir(folder_or_tiff):
            self.file_path = folder_or_tiff
            return self._dry_run_estimate()
        else:
            self.file_path = folder_or_tiff
            return self._dry_run_estimate()

    @staticmethod
    def _infer_mode(path: str) -> str:
        if path.lower().endswith((".csv",)):
            return "CSV"
        if os.path.isdir(path):
            return "FOLDER"
        if path.lower().endswith((".tif", ".tiff")):
            return "TIFF"
        raise ValueError("Path must be .csv, a folder, or a .tif/.tiff file")

    @staticmethod
    def _default_output_dir(path: str) -> str:
        if os.path.isdir(path):
            return os.path.join(path, "_preprocess_ui_logs")
        else:
            base = os.path.dirname(path)
            return os.path.join(base, "_preprocess_ui_logs")

    @staticmethod
    def _compute_max_level(shape, min_size=100) -> int:
        # mirror of compute_maximum_level using mean
        import numpy as np
        z, y, x = shape
        base_size = np.mean([z, y, x])
        level = 0
        while base_size >= min_size * 2:
            base_size //= 2
            level += 1
        return level


# ------------------------ Main Window ------------------------ #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Preprocessing UI")
        self.resize(920, 680)
        self.worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Input Group ---
        g_input = QGroupBox("Input Source")
        grid = QGridLayout(g_input)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # row 0: Input mode (Radio Buttons)
        self.mode_csv = QRadioButton("CSV")
        self.mode_dir = QRadioButton("Folder (TIFF Series)")
        self.mode_tiff = QRadioButton("Single 3D TIFF")
        self.mode_csv.setChecked(True)
        grp = QButtonGroup(self)  # For exclusivity
        grp.addButton(self.mode_csv)
        grp.addButton(self.mode_dir)
        grp.addButton(self.mode_tiff)
        grid.addWidget(QLabel("Mode:"), 0, 0)
        grid.addWidget(self.mode_csv, 0, 1)
        grid.addWidget(self.mode_dir, 0, 2)
        grid.addWidget(self.mode_tiff, 0, 3)

        # row 1: path
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("Select .csv / Folder / Single 3D TIFF")
        self.edit_path.setMinimumWidth(300)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self.browse_input)
        grid.addWidget(QLabel("Path:"), 1, 0)
        grid.addWidget(self.edit_path, 1, 1)
        grid.addWidget(btn_browse, 1, 2)

        # row 2: dtype
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["Auto (from data)", "uint16", "float32"])
        grid.addWidget(QLabel("Data Type (dtype):"), 2, 0)
        grid.addWidget(self.dtype_combo, 2, 1)

        # row 3: LOD output folder name
        self.lod_name_edit = QLineEdit()
        self.lod_name_edit.setText("LOD_Data")
        grid.addWidget(QLabel("Output Folder Name (LOD):"), 3, 0)
        grid.addWidget(self.lod_name_edit, 3, 1)

        # row 4: min_size
        self.min_size = QSpinBox()
        self.min_size.setRange(16, 4096)
        self.min_size.setValue(100)
        grid.addWidget(QLabel("min_size (for LOD estimation):"), 4, 0)
        grid.addWidget(self.min_size, 4, 1)

        root.addWidget(g_input)

        # --- Actions ---
        row = QHBoxLayout()
        self.btn_estimate = QPushButton("Dry-Run (Estimate Levels)")
        self.btn_start = QPushButton("Start Preprocessing")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        row.addWidget(self.btn_estimate)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_cancel)
        root.addLayout(row)

        # --- Progress + Stage ---
        self.stage_label = QLabel("Not Started")
        self.bar = QProgressBar();
        self.bar.setRange(0, 100)
        root.addWidget(self.stage_label)
        root.addWidget(self.bar)
        # --- Log Pane ---
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        # wire buttons
        self.btn_estimate.clicked.connect(self.on_estimate)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)

    # ---------- UI Slots ---------- #
    def browse_input(self):
        if self.mode_csv.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv)")
        elif self.mode_tiff.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Select 3D TIFF", "", "TIFF (*.tif *.tiff)")
        else:  # Folder mode
            path = QFileDialog.getExistingDirectory(self, "Select Folder (containing 2D TIFF series)")
        if path:
            self.edit_path.setText(path if isinstance(path, str) else path[0])

    def on_estimate(self):
        path = self.edit_path.text().strip()
        if not path:
            return QMessageBox.warning(self, "Warning", "Please select an input path first")
        self._start_worker(path, dry_run=True)

    def on_start(self):
        path = self.edit_path.text().strip()
        if not path:
            return QMessageBox.warning(self, "Warning", "Please select an input path first")
        self._start_worker(path, dry_run=False)

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_abort()  # placeholder; actual cooperative cancel requires pipeline support
            QMessageBox.information(
                self, "Cancelling",
                "Cancellation request has been sent (current version will stop after finishing this stage)"
            )

    # ---------- Worker wiring ---------- #
    def _start_worker(self, path: str, dry_run: bool):
        if self.worker and self.worker.isRunning():
            return QMessageBox.information(self, "Busy", "Processing not finished yet. Please wait or press Cancel.")

        self.log.clear()
        self.bar.setValue(0)
        self.stage_label.setText("Scheduling…")

        log_dir = Worker._default_output_dir(path)
        os.makedirs(log_dir, exist_ok=True)
        self.ui_log_path = os.path.join(log_dir, "ui_log.txt")

        try:
            mode = "CSV" if self.mode_csv.isChecked() else "FOLDER" if self.mode_dir.isChecked() else "TIFF"
            header = (
                    "\n" + "=" * 80 + "\n"
                                      f"[Session Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                      f"Path: {path}\nMode: {mode}\nDType: {self.dtype_combo.currentText()}\n"
                                      f"LOD Name: {self.lod_name_edit.text().strip() or 'LOD_Data'}\n"
                                      f"Dry-Run: {dry_run}\n"
                    + "=" * 80 + "\n"
            )
            with open(self.ui_log_path, "a", encoding="utf-8") as f:
                f.write(header)
        except Exception as e:
            self.log.appendPlainText(f"[Warn] Cannot write ui_log.txt: {e}")


        self.worker = Worker(
            file_path=path,
            dtype_name=self.dtype_combo.currentText(),
            dry_run=dry_run,
            custom_min_size=int(self.min_size.value()),
            lod_name=self.lod_name_edit.text().strip() or "LOD_Data"  # ✅ Pass along
        )
        self.worker.started_human.connect(lambda msg: self._append_log(msg))
        self.worker.stage.connect(self._on_stage)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self.bar.setValue)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)

        self.btn_start.setEnabled(False)
        self.btn_estimate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker.start()

    def _on_stage(self, s: str):
        self.stage_label.setText(s)
        self._append_log(f"[Stage] {s}")

    def _append_log(self, s: str):
        self.log.appendPlainText(s)
        if self.ui_log_path:
            try:
                with open(self.ui_log_path, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
            except Exception as e:
                self.log.appendPlainText(f"[Warn] Failed to append ui_log.txt: {e}")

    def _on_done(self):
        self.btn_start.setEnabled(True)
        self.btn_estimate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.stage_label.setText("✔ Completed")
        self.bar.setValue(100)
        if self.ui_log_path:
            try:
                from datetime import datetime
                with open(self.ui_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[Session End] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception as e:
                self.log.appendPlainText(f"[Warn] Failed to finalize ui_log.txt: {e}")
        QMessageBox.information(self, "Completed", "Preprocessing has finished successfully!")

    def _on_failed(self, err: str):
        self.btn_start.setEnabled(True)
        self.btn_estimate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.stage_label.setText("❌ Failed")
        self.bar.setValue(0)
        self._append_log(err)
        QMessageBox.critical(self, "Error", f"Processing failed:\n\n{err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
