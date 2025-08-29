"""
LV-Vis Launcher (PyQt5)
A simple GUI to launch the LV-Vis multi-volume LOD viewer (LVVisWindow).

What this does
- Lets you pick input data (single folder or multi-volume from CSV / multiple folders).
- Validates inputs and (optionally) creates a temporary CSV for you.
- Lets you set the viewer window size (Width/Height; default 1200 × 1200).
- Opens the main LV-Vis window (`LVVisWindow`) with your selections.

Key features
- Two modes:
  1) Single Volume → select one folder
  2) Multi-Volume  → (A) From CSV  or  (B) From Multiple Folders + Manual Coordinates
- CSV validator (no header; exactly 4 columns: `path,x,y,z`).
- Auto-generate a temporary CSV when launching from multiple folders + coordinates.
- Adjustable LOD window size via spin boxes (400–8000 px, step 100, default 1200).

How to run
    python lv_vis_gui.py

Usage
1) Single Volume
   - Click “Browse Folder”, pick a volume folder, then “Launch”.
   - The launcher will create a temporary CSV with `[folder,0,0,0]`.

2) Multi-Volume (From CSV)
   - Click “Browse CSV”, select a 4-column CSV with rows: `path,x,y,z` (no header).
   - Click “Launch”.

3) Multi-Volume (From Multiple Folders + Manual Coordinates)
   - Click “Add Folder” to add rows; edit X/Y/Z (integers).
   - “Launch” will generate a temporary CSV and open the viewer.

CSV format
- No header; exactly 4 columns in this order:
    path, x, y, z
- Coordinates are in original-resolution voxel units.
- Internally the viewer stores positions as (Z, Y, X).

Window size
- “LOD Window Size” lets you set width/height (px). Default is 1200 × 1200.
- The values are passed into `LVVisWindow(csv_path, w=..., h=...)`.

Assumptions
- `lv_vis.py` exports `LVVisWindow`.
- PyQt5 is installed; VisPy and the LV-Vis project dependencies are available.
- The selected folders/CSV paths are accessible on this machine.

Notes
- The temporary CSVs are written to your OS temp directory and removed by you if needed.
- If launching fails, re-check the CSV format, folder paths, and console messages.
"""

import os, sys, tempfile, csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView
)
from PyQt5.QtCore import Qt
from lv_vis import LVVisWindow


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Data Loading Mode")
        self.resize(760, 520)

        root = QWidget()
        self.setCentralWidget(root)
        lay  = QVBoxLayout(root)

        # ---- Mode Selection ----
        mode_box = QGroupBox("Mode")
        g = QHBoxLayout(mode_box)
        self.rb_single = QRadioButton("Single Volume")
        self.rb_multi  = QRadioButton("Multi-Volume")
        self.rb_single.setChecked(True)
        g.addWidget(self.rb_single); g.addWidget(self.rb_multi)
        lay.addWidget(mode_box)

        # ---- Single Volume: one folder ----
        self.single_box = QGroupBox("Single Volume: Select a Folder")
        grid1 = QGridLayout(self.single_box)
        self.single_dir = QLineEdit()
        btn_single_browse = QPushButton("Browse Folder")
        btn_single_browse.clicked.connect(self._pick_single_dir)
        grid1.addWidget(QLabel("Folder:"), 0, 0)
        grid1.addWidget(self.single_dir, 0, 1)
        grid1.addWidget(btn_single_browse, 0, 2)
        lay.addWidget(self.single_box)

        # ---- Multi-Volume: two sources (CSV / Multiple Folders) ----
        self.multi_box = QGroupBox("Multi-Volume")
        grid2 = QGridLayout(self.multi_box)

        # 2A) CSV
        self.rb_multi_csv   = QRadioButton("From CSV")
        self.rb_multi_table = QRadioButton("From Multiple Folders + Manual Coordinates")
        self.rb_multi_csv.setChecked(True)
        grp2 = QButtonGroup(self)
        grp2.addButton(self.rb_multi_csv); grp2.addButton(self.rb_multi_table)

        # CSV row
        self.csv_path = QLineEdit()
        btn_csv_browse = QPushButton("Browse CSV")
        btn_csv_browse.clicked.connect(self._pick_csv)
        grid2.addWidget(self.rb_multi_csv, 0, 0)
        grid2.addWidget(self.csv_path, 0, 1)
        grid2.addWidget(btn_csv_browse, 0, 2)

        # 2B) Multiple folders table
        grid2.addWidget(self.rb_multi_table, 1, 0, 1, 3)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Path", "X", "Y", "Z"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        grid2.addWidget(self.table, 2, 0, 1, 3)

        row_btns = QHBoxLayout()
        btn_add  = QPushButton("Add Folder")
        btn_del  = QPushButton("Delete Selected Rows")
        btn_clear= QPushButton("Clear All")
        btn_add.clicked.connect(self._add_folder_row)
        btn_del.clicked.connect(self._remove_selected_rows)
        btn_clear.clicked.connect(self._clear_table)
        row_btns.addWidget(btn_add); row_btns.addWidget(btn_del); row_btns.addWidget(btn_clear)
        grid2.addLayout(row_btns, 3, 0, 1, 3)

        lay.addWidget(self.multi_box)

        # ---- LOD Window Size ----
        size_box = QGroupBox("LOD Window Size")
        sz = QGridLayout(size_box)

        from PyQt5.QtWidgets import QSpinBox
        self.spin_w = QSpinBox()
        self.spin_h = QSpinBox()
        for sp in (self.spin_w, self.spin_h):
            sp.setRange(400, 8000)
            sp.setSingleStep(100)
            sp.setValue(1200)  # default 1200

        sz.addWidget(QLabel("Width (px):"), 0, 0)
        sz.addWidget(self.spin_w, 0, 1)
        sz.addWidget(QLabel("Height (px):"), 1, 0)
        sz.addWidget(self.spin_h, 1, 1)

        lay.addWidget(size_box)

        # ---- Bottom Actions ----
        bottom = QHBoxLayout()
        self.btn_launch = QPushButton("Launch")
        self.btn_launch.clicked.connect(self._launch)
        bottom.addStretch(1); bottom.addWidget(self.btn_launch)
        lay.addLayout(bottom)

        # toggle single vs multi
        self.rb_single.toggled.connect(self._refresh_mode)
        self._refresh_mode()

    # ---------- UI helpers ----------
    def _refresh_mode(self):
        is_single = self.rb_single.isChecked()
        self.single_box.setEnabled(is_single)
        self.multi_box.setEnabled(not is_single)

    def _pick_single_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Single Volume Folder")
        if d:
            self.single_dir.setText(d)

    def _pick_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv)")
        if f:
            self.csv_path.setText(f)

    def _add_folder_row(self):
        d = QFileDialog.getExistingDirectory(self, "Select a Volume Folder")
        if not d:
            return
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(d))
        for c in (1, 2, 3):
            item = QTableWidgetItem("0")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, c, item)

    def _remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _clear_table(self):
        self.table.setRowCount(0)

    # ---------- Launch logic ----------
    def _launch(self):
        if self.rb_single.isChecked():
            folder = self.single_dir.text().strip()
            if not folder:
                return QMessageBox.warning(self, "Warning", "Please select a folder")
            if not os.path.isdir(folder):
                return QMessageBox.warning(self, "Warning", "Path is not a folder")
            csv_path = self._make_temp_csv_single(folder)
            return self._open_lod(csv_path)

        # Multi
        if self.rb_multi_csv.isChecked():
            csv_path = self.csv_path.text().strip()
            if not csv_path:
                return QMessageBox.warning(self, "Warning", "Please select a CSV file")
            if not os.path.isfile(csv_path):
                return QMessageBox.warning(self, "Warning", "CSV file not found")
            if not self._validate_csv_4cols(csv_path):
                return
            return self._open_lod(csv_path)

        # Multi with folders + coords
        rows = self.table.rowCount()
        if rows == 0:
            return QMessageBox.warning(self, "Warning", "Please add folders and fill in coordinates")
        # Validate & generate temp CSV
        data = []
        for r in range(rows):
            path = self._item_text(r, 0)
            try:
                x = int(self._item_text(r, 1))
                y = int(self._item_text(r, 2))
                z = int(self._item_text(r, 3))
            except ValueError:
                return QMessageBox.warning(self, "Warning", f"Row {r+1}: coordinates must be integers")
            if not path or not os.path.isdir(path):
                return QMessageBox.warning(self, "Warning", f"Row {r+1}: invalid path")
            data.append((path, x, y, z))
        csv_path = self._make_temp_csv_multi(data)
        return self._open_lod(csv_path)

    def _item_text(self, row, col):
        it = self.table.item(row, col)
        return it.text().strip() if it else ""

    # Temporary CSV: single volume (path only; coords default to 0,0,0)
    def _make_temp_csv_single(self, folder):
        fd, path = tempfile.mkstemp(prefix="single_", suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([folder, 0, 0, 0])
        return path

    # Temporary CSV: multi-volume with coords (each row: path,x,y,z; no header)
    def _make_temp_csv_multi(self, rows):
        fd, path = tempfile.mkstemp(prefix="multi_", suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            for (p, x, y, z) in rows:
                w.writerow([p, x, y, z])
        return path

    # Open LOD window
    def _open_lod(self, csv_path: str):
        self.hide()
        w = int(self.spin_w.value())
        h = int(self.spin_h.value())
        self.viewer = LVVisWindow(csv_path, w=w, h=h)
        self.viewer.show()

    def _validate_csv_4cols(self, csv_path: str) -> bool:
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            for i, row in enumerate(r, start=1):
                if len(row) != 4:
                    QMessageBox.warning(self, "Format Error", f"Line {i} does not have 4 columns [path,x,y,z]")
                    return False
                try:
                    float(row[1]); float(row[2]); float(row[3])
                except Exception:
                    QMessageBox.warning(self, "Format Error", f"Line {i} coordinates are not numeric: {row[1:]}")
                    return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())
