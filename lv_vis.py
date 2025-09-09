# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh

"""
LV-Vis Viewer (PyQt5 + VisPy)
A lightweight launcher/window for multi-volume LOD visualization.

What it is
- Standalone viewer for octree/LOD volume datasets with two canvases:
  an Interactive view and a Global/overview view. :contentReference[oaicite:0]{index=0}

Key features
- Flexible input: CSV (rows = path,x,y,z; no header) or a single folder
  (auto-generates a temporary CSV with coords). :contentReference[oaicite:1]{index=1}
- Adjustable window size via CLI args or the GUI launcher (defaults 1200×1200). :contentReference[oaicite:2]{index=2}
- Per-volume world positions (stored internally as Z,Y,X), per-level rendering, and global overview box.
- Camera tools: azimuth/elevation/roll, “Center Camera”, interactive screenshot, ROI picking box.
- Memory HUD: live GPU VRAM & Python RSS charts, optional CSV logging.

How to run (CLI)
    # CSV (no header; 4 cols: path,x,y,z)
    python lv_vis.py --csv ./datapath_csv/my_vols.csv --width 1600 --height 1000

    # Single folder (generate temp CSV) with world coords
    python lv_vis.py --folder /data/rod1/part3_1 --coords 0 0 0 --width 1400 --height 1000

    # Legacy positional arg (kept for compatibility)
    python lv_vis.py ./datapath_csv/rod_multi_1vol.csv

CSV format
- No header; exactly 4 columns: path, x, y, z (original-resolution voxel units).
- Internally stored as (Z, Y, X) during loading and alignment.

Assumptions
- PyQt5 and VisPy are installed and working.
- Project modules (e.g., LVVisWindow, controller/loader classes) are importable.
- If you prefer a GUI launcher for picking CSV/folders and window size,
  run the launcher script (lv_vis_gui.py) and press “Launch”. :contentReference[oaicite:3]{index=3}

Notes
- Window width/height passed via CLI or the launcher are forwarded to LVVisWindow.
- ROI and picking visuals operate in (X,Y,Z) screen/world order; stored box
  bounds and global bookkeeping use (Z,Y,X) for consistency with volume data.
"""


import sys
import os
import numpy as np
from vispy import scene, app
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore    import QTimer, Qt
from src.canvas_setting import CanvasManager, CameraController, AxisVisualizer
from src.vol_setting import MultiVolumeController
from src.select_box import VolumeSelector
from src.control_panel import ControlPanel
import argparse, tempfile, csv
import time
import psutil

class LVVisWindow(QMainWindow):
    def __init__(self, csv_path: str, w: int = 1200, h: int = 1200):
        super().__init__()
        self.setWindowTitle("LV-Vis Viewer")

        self.w, self.h = int(w), int(h)
        self.setGeometry(100, 100, self.w, self.h)
        # build central widget
        main_w   = QWidget()
        self.setCentralWidget(main_w)

        main_lay = QVBoxLayout(main_w)

        # setting widget for canvas
        canvas_container = QWidget()
        canvas_lay = QHBoxLayout(canvas_container)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.setSpacing(20)

        # setting Interactive canvas + title
        #####################################
        itac_col = QWidget()
        itac_lay = QVBoxLayout(itac_col)
        itac_lay.setContentsMargins(0, 0, 0, 0)
        itac_lay.setSpacing(10)

        itac_title = QLabel("Interactive Canvas")
        itac_title.setStyleSheet("font: bold 24px; color: black;")
        itac_title.setAlignment(Qt.AlignCenter)
        itac_lay.addWidget(itac_title)

        # use CanvasManager to control Interactive canvas
        self.canvas_manager = CanvasManager(main_w, self.w, self.h)
        self.view = self.canvas_manager.view
        self.camera_controller = CameraController(self.view)
        self.axis_vis = AxisVisualizer(self.view, self.camera_controller.camera)
        self.canvas_manager.canvas.native.setFixedSize(self.w, self.h)
        itac_lay.addWidget(self.canvas_manager.canvas.native)
        #####################################

        # setting global canvas + title
        #####################################
        glo_col = QWidget()
        glo_lay = QVBoxLayout(glo_col)
        glo_lay.setContentsMargins(0, 0, 0, 0)
        glo_lay.setSpacing(10)

        glo_title = QLabel("Global Canvas")
        glo_title.setStyleSheet("font: bold 24px; color: black;")
        glo_title.setAlignment(Qt.AlignCenter)
        glo_lay.addWidget(glo_title)

        # use CanvasManager to control global canvas
        self.global_canvas_manager = CanvasManager(main_w, self.w, self.h)
        self.global_view  = self.global_canvas_manager.view
        self.global_camera_controller = CameraController(self.global_view)
        self.global_axis_vis  = AxisVisualizer(self.global_view, self.global_camera_controller.camera)
        self.global_canvas_manager.canvas.native.setFixedSize(self.w, self.h)
        glo_lay.addWidget(self.global_canvas_manager.canvas.native)
        #####################################

        # add 2 canvas together
        canvas_lay.addWidget(itac_col)
        canvas_lay.addWidget(glo_col)

        main_lay.addWidget(canvas_container)

        #  Multi‐Volume Controller setting
        # factors = [32, 16, 8, 4, 2, 1]
        self.controller = MultiVolumeController(
            view=self.view,
            gView=self.global_view,
            # factors=factors
        )
        self.controller.add_volume(csv_path)

        base_shape = self.controller.shape
        self.selector = VolumeSelector(
            self.canvas_manager.canvas,
            self.view,
            base_shape,
            self.w,
            self.h
        )
        self.ctrl_pressed   = False
        self.pending_click  = None
        self.selected_index = 0

        # moving and rotation
        self.move_step      = 2.0
        self.is_rotating    = False
        self.rotation_speed = 0.5
        self.rotation_timer = app.Timer('auto',
                                        connect=self.rotate_camera,
                                        start=False)

        # ControlPanel（
        first_loader = next(iter(self.controller.volumes.values()))
        vol0_data    = first_loader.volumes[0]   # level 0 的 numpy array
        self.ctrl_panel = ControlPanel(vol0_data, parent=self, main_window=self)
        self.ctrl_panel.populate_volumes(self.controller.volumes)
        main_lay.addWidget(self.ctrl_panel)

        self._recenter_camera_on_volumes()
        self.camera_controller.camera.center += np.array(self.controller.volumes[0].volumes[0].shape)[::-1] / 2

        # bind canvas events
        canvas = self.canvas_manager.canvas
        canvas.events.mouse_press.connect(self.on_mouse_press)
        canvas.events.draw      .connect(self.on_draw)
        canvas.events.key_press .connect(self.on_key_press)
        canvas.events.key_release.connect(self.on_key_release)

        self.show()


    def rotate_camera(self, event):
        if self.is_rotating:
            self.camera_controller.camera.azimuth += self.rotation_speed
            self.canvas_manager.canvas.update()

    def _recenter_camera_on_volumes(self):
        translates = []
        for ldr in self.controller.volumes.values():
            if ldr.volume_visuals.parent:
                tr = ldr.volume_visuals.transform.translate  # tuple (x,y,z)
                translates.append(np.array(tr, dtype=float))
        avg = np.mean(np.stack(translates, axis=0), axis=0)
        print(f"[_recenter_camera] Average volume visual translate = {avg}")

        self.camera_controller.camera.center = tuple(avg[:3])
        self.canvas_manager.canvas.update()
        self.ctrl_panel.refresh_current()

    def on_key_press(self, event):
        lvl = self.controller.current_layer
        loaders = list(self.controller.volumes.values())

        if event.key == 'Alt' and lvl > 0:
            self._reload_layer()
            return

        if event.key == 'Control':
            self.ctrl_pressed = True
            return

        # Moving ROI
        if self.ctrl_pressed and self.selector.voxel is not None:
            delta = np.zeros(3, dtype=int)
            if event.key == 'A':
                delta[0] = -1  # X - 1
            elif event.key == 'D':
                delta[0] = +1  # X + 1
            elif event.key == 'W':
                delta[1] = +1  # Y - 1
            elif event.key == 'S':
                delta[1] = -1  # Y + 1
            elif event.key == 'Q':
                delta[2] = -1  # Z - 1
            elif event.key == 'E':
                delta[2] = +1  # Z + 1
            else:
                return

            self.selector.move_box(delta)
            self.canvas_manager.canvas.update()
            return

        # Zoom in on press 'Y'
        if event.key == 'Y' and self.selector.box_start is not None:
            print(f"selector box start: {self.selector.box_start}")
            self._drill_overlaps(threshold=0.3)
            self.selector.remove_box()
            globox_in_ldr = []
            for ldr in loaders:
                globox_in_ldr.append([ldr.vol_global_start_point[self.controller.current_layer]+ldr.raw_positions,
                                      ldr.vol_global_end_point[self.controller.current_layer]+ldr.raw_positions])

            self.selector.draw_box_in_global(self.global_view, loaders[0].factor, globox_in_ldr, lvl)
            self.global_canvas_manager.canvas.update()
            return
        # Cancel ROI selection on 'N'
        if event.key == 'N':
            self.selector.remove_box()
            self.canvas_manager.canvas.update()
            return

        # Toggle rotation on 'R'
        if event.key == 'R':
            self.is_rotating = not self.is_rotating
            (self.rotation_timer.start() if self.is_rotating else self.rotation_timer.stop())
            return

        # Switch selected volume index for volume moving
        if event.key == 'O':
            self.selected_index = max(0, self.selected_index - 1)
            self._highlight_selected()
            return
        if event.key == 'P':
            self.selected_index = min(
                len(loaders) - 1,
                self.selected_index + 1
            )
            self._highlight_selected()
            return

        # WASDQE: move current volume
        if event.key.name in ['A', 'D', 'W', 'S', 'Q', 'E']:
            loader = list(self.controller.volumes.values())[self.selected_index]
            factor = loader.factor[lvl]
            delta = np.zeros(3, dtype=float)
            if event.key == 'A':
                delta[0] = -self.move_step # X - move_step
            elif event.key == 'D':
                delta[0] = +self.move_step # X + move_step
            elif event.key == 'W':
                delta[1] = +self.move_step # Y - move_step
            elif event.key == 'S':
                delta[1] = -self.move_step # Y + move_step
            elif event.key == 'Q':
                delta[2] = -self.move_step # Z - move_step
            elif event.key == 'E':
                delta[2] = +self.move_step # Z + move_step
            loader.translations += delta

            # Calculate local translate
            raw = loader.raw_positions + loader.translations + loader.vol_global_start_point[lvl]

            local = (raw // factor)[::-1]
            print(f"volume index: {self.selected_index} raw positions: {raw}, local: {local}")

            vis = loader.volume_visuals
            tr = vis.transform
            tr.translate = tuple(local)
            vis.transform = tr

            posVis = loader.volume_pos_visuals
            trpos = posVis.transform
            trpos.translate = tuple(local)
            posVis.transform = trpos

            rawglo = loader.raw_positions + loader.translations
            gloVis = loader.gloVolume_visuals
            trglo = gloVis.transform
            trglo.translate = tuple((rawglo//loader.factor[0])[::-1])
            gloVis.transform = trglo

            self.global_canvas_manager.canvas.update()
            self.canvas_manager.canvas.update()
            self.ctrl_panel.refresh_current()
            return

    def on_key_release(self, event):
        if event.key == 'Control':
            self.ctrl_pressed = False

    def on_mouse_press(self, event):
        if not self.ctrl_pressed:
            return
        loaders = list(self.controller.volumes.values())

        if all(self.controller.current_layer < ldr.max_level for ldr in loaders):
            ps = self.canvas_manager.canvas.pixel_scale
            x = int(event.pos[0] * ps)
            y = int(event.pos[1] * ps)
            self.pending_click = (x, y)
            self.canvas_manager.canvas.update()
        else:
            print(f'Reached maximum LOD depth (current={self.controller.current_layer})')  # ★ 順便補上右括號
            return

    def on_draw(self, event):
        if not self.ctrl_pressed or self.pending_click is None:
            return
        x, y = self.pending_click
        self.pending_click = None

        lvl = self.controller.current_layer
        loaders = list(self.controller.volumes.values())
        visuals = [ldr.volume_visuals for ldr in loaders]
        pos_visuals = [ldr.volume_pos_visuals for ldr in loaders]
        factor = loaders[0].factor[lvl]

        shapes = [(ldr.vol_size).astype(int) for ldr in loaders]
        print(f'[on_draw] shapes', shapes)
        translates = [np.array(vis.transform.translate) for vis in visuals]
        print(f'[on_draw] translates', translates)

        ps = self.canvas_manager.canvas.pixel_scale
        fb_w = int(self.w * ps)
        fb_h = int(self.h * ps)

        idx, vec = self.selector.handle_click(
            x, y, fb_w, fb_h,
            pos_visuals=pos_visuals,
            shapes=shapes,
            transforms=translates
        )
        # vec = shapes[0]/2
        # print("shape/2 ",vec)
        print(f"idx: {idx}, vec: {vec}")
        if idx is not None:
            t = np.array(
                loaders[idx].raw_positions + loaders[idx].translations + loaders[idx].vol_global_start_point[lvl])
            self.selector.draw_box(vec, t / factor)

        self.canvas_manager.canvas.update()

    def _highlight_selected(self):
        lvl = self.controller.current_layer
        loaders = list(self.controller.volumes.values())
        print(self.selected_index)
        for i, ldr in enumerate(loaders):
            gamma = 0.5 if i == self.selected_index else 1.0
            ldr.volume_visuals.gamma = gamma
        self.canvas_manager.canvas.update()
        app.Timer(1.0, connect=self._reset_gamma, start=True)

    def _reset_gamma(self, event=None):
        for ldr in self.controller.volumes.values():
            ldr.volume_visuals.gamma = 1.0
        self.canvas_manager.canvas.update()

    def _drill_overlaps(self, threshold: float = 0.25):
        t0 = time.perf_counter()
        m0 = _mem_snapshot()
        any_overlap = False
        union_gs = None
        union_ge = None
        try:
            lvl = self.controller.current_layer
            loaders = list(self.controller.volumes.values())

            # --- Step 1: Extract box_s, box_e (layer L voxel index) ---
            box_s_layer = self.selector.box_start  # np.ndarray([z, y, x])
            box_e_layer = self.selector.box_end  # np.ndarray([z, y, x])
            print("\n=== [_drill_overlaps] Level =", lvl)
            print(f"[_drill_overlaps] 1) User ROI box (Z,Y,X) Layer-{lvl} index → "
                  f"box_s_layer = {box_s_layer}, box_e_layer = {box_e_layer}")

            any_overlap = False  # Set True if at least one loader overlaps

            for i, ldr in enumerate(loaders):
                if ldr.volume_visuals:
                    ldr.volume_visuals.parent = None
                print(f"[_drill_overlaps] \n--- Processing loader id = {ldr.id} (#{i}) ---")
                volMaximum = ldr.vol_size * ldr.factor[0]
                factor = ldr.factor[lvl]
                print(f'[_drill_overlaps] Factor = {factor}')

                # --- Step 2: Convert box indices at layer L → global voxel coords at original resolution ---
                box_s_global = box_s_layer * factor
                box_e_global = box_e_layer * factor
                print(f"[_drill_overlaps] 2) box_s_global (Layer {lvl} → global coords) = {box_s_global}")
                print(f"[_drill_overlaps]    box_e_global (Layer {lvl} → global coords) = {box_e_global}")

                # --- Step 3: Compute this loader's global range at original resolution ---
                raw_global = ldr.vol_global_start_point[lvl] + (ldr.raw_positions + ldr.translations)
                print(f"[_drill_overlaps] 3) raw_global (Layer {lvl}, loader {ldr.id} "
                      f"= start_point + raw_positions + translations) = {raw_global}")
                print(ldr.vol_global_start_point[lvl], ldr.raw_positions, ldr.translations)

                vol_s_global = raw_global.copy()
                vol_e_global = np.minimum(raw_global + ldr.vol_size * factor,
                                          volMaximum + ldr.raw_positions)
                print(f"[_drill_overlaps]    vol_s_global (global start) = {vol_s_global}")
                print(f"[_drill_overlaps]    vol_e_global (global end)   = {vol_e_global}")

                # --- Step 4: Check overlap in global voxel coordinates ---
                ov_s_global = np.maximum(vol_s_global, box_s_global)
                ov_e_global = np.minimum(vol_e_global, box_e_global)
                print(f"[_drill_overlaps] 4) ov_s_global = max(vol_s_global, box_s_global) = {ov_s_global}")
                print(f"[_drill_overlaps]    ov_e_global = min(vol_e_global, box_e_global) = {ov_e_global}")

                if np.all(ov_s_global < ov_e_global):
                    any_overlap = True
                    print("[_drill_overlaps]    → Overlap detected. Proceeding to next-level extraction")

                    # --- Step 5: Compute extraction center (center_local) ---
                    mid_global = ov_s_global + ((ov_e_global - ov_s_global) // 2)
                    print(f"[_drill_overlaps] 5) mid_global (intersection center, global) = {mid_global}")

                    mid_layer = (mid_global // factor).astype(int)
                    print(f"[_drill_overlaps]    mid_layer = {mid_layer} (Layer {lvl} voxel index)")

                    local_offset = ((ldr.raw_positions + ldr.translations +
                                     ldr.vol_global_start_point[lvl]) // factor).astype(int)
                    print(f"[_drill_overlaps]    local_offset = "
                          f"(raw_positions + translations + vol_global_start_point) // factor = {local_offset}")

                    center_local = (mid_layer - local_offset).astype(int)
                    print(f"[_drill_overlaps] 6) center_local (coordinate for extract_next) = {center_local}")

                    subvol = ldr.extract_next(center_local, lvl)
                    print(f"[_drill_overlaps] 7) Extracted subvol, shape = {subvol.shape}")

                    ldr.render_level(lvl + 1, self.view, threshold=threshold)
                else:
                    print("[_drill_overlaps]    → No overlap, skipping this loader")

            # --- Final Step: If any overlap found, update layer and redraw ---
            if any_overlap:
                # self.selector.set_actual_roi_global(lvl, union_gs, union_ge)

                print(f"\n[_drill_overlaps] Updating controller.current_layer: {lvl} → {lvl + 1}\n")
                self.controller.current_layer += 1
                self._recenter_camera_on_volumes()
                self.canvas_manager.canvas.update()
                self.ctrl_panel.refresh_current()
            else:
                print("\n[_drill_overlaps] >>> No overlaps. Layer not updated, no redraw <<<\n")

        finally:
            t1 = time.perf_counter()
            m1 = _mem_snapshot()
            _print_perf_report("drill_overlaps", t0, m0, t1, m1)

    def _reload_layer(self):
        t0 = time.perf_counter()
        m0 = _mem_snapshot()
        try:
            self.controller.current_layer -= 1
            lvl = self.controller.current_layer

            self.selector._removeGlobalBox(lvl)
            self.selector._reload_global(self.global_view, lvl - 1)

            loaders = list(self.controller.volumes.values())

            for i, ldr in enumerate(loaders):
                if ldr.has_level(lvl + 1):  # 建議加一個 helper
                    print(f'[_reload_layer] free level {lvl + 1} for volume {i}')
                    ldr.free_level(lvl + 1)

            for i, ldr in enumerate(loaders):
                if ldr.has_level(lvl):
                    print(f'[_reload_layer] render level {lvl} for volume {i}')
                    ldr.render_level(lvl, self.view)

            self._recenter_camera_on_volumes()
            self.ctrl_panel.refresh_current()

        finally:
            t1 = time.perf_counter()
            m1 = _mem_snapshot()
            _print_perf_report("reload_layer", t0, m0, t1, m1)

def _mem_snapshot():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    rss_mb = mi.rss / (1024 ** 2)
    vms_mb = mi.vms / (1024 ** 2)
    gpu_used_mb = None
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
        except Exception:
            pass
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used_mb = mem.used / (1024 ** 2)
    except Exception:
        gpu_used_mb = None

    return {"rss_mb": rss_mb, "vms_mb": vms_mb, "gpu_mb": gpu_used_mb}

def _print_perf_report(label, t0, m0, t1, m1):
    dt = (t1 - t0) * 1000.0  # ms
    drss = m1["rss_mb"] - m0["rss_mb"]
    dvms = m1["vms_mb"] - m0["vms_mb"]
    if (m0["gpu_mb"] is not None) and (m1["gpu_mb"] is not None):
        dgpu = m1["gpu_mb"] - m0["gpu_mb"]
        gpu_line = f", GPU Δused = {dgpu:+.1f} MB (→ {m1['gpu_mb']:.1f} MB)"
    else:
        gpu_line = ""
    print(
        f"[PERF] {label}: {dt:.1f} ms | RAM ΔRSS = {drss:+.1f} MB (→ {m1['rss_mb']:.1f} MB), "
        f"ΔVMS = {dvms:+.1f} MB (→ {m1['vms_mb']:.1f} MB){gpu_line}"
    )



def _make_temp_csv_single(folder: str, coords=(0, 0, 0)) -> str:
    fd, path = tempfile.mkstemp(prefix="single_", suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        x, y, z = coords
        w.writerow([folder, x, y, z])
    return path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Launch LV-Vis Viewer (CSV or single folder) with adjustable window size."
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="CSV file with rows: path,x,y,z")
    src.add_argument("--folder", help="Single volume folder (will make a temp CSV)")
    p.add_argument("legacy_csv", nargs="?", help="(legacy) CSV path as positional arg")

    # coors
    p.add_argument("--coords", nargs=3, type=int, metavar=("X", "Y", "Z"),
                   default=(0, 0, 0),
                   help="Raw position for --folder mode (default: 0 0 0)")

    # window size
    p.add_argument("--width", type=int, default=1200, help="LOD window width (px)")
    p.add_argument("--height", type=int, default=1200, help="LOD window height (px)")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.csv:
        csv_path = args.csv
    elif args.folder:
        csv_path = _make_temp_csv_single(args.folder, coords=tuple(args.coords))
    elif args.legacy_csv:
        csv_path = args.legacy_csv
    else:
        csv_path = './datapath_csv/rod_multi_1vol.csv'

    app_qt = QApplication(sys.argv)
    window = LVVisWindow(csv_path, w=args.width, h=args.height)
    window.show()
    sys.exit(app_qt.exec_())

