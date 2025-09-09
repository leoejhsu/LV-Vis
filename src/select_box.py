# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh

from vispy.scene.visuals import Box
from vispy.visuals.transforms import STTransform
from vispy import gloo
import numpy as np


class VolumeSelector:
    def __init__(self, canvas, view, shape, w, h):
        self.canvas = canvas
        self.shape = shape
        self.view = view

        ps = self.canvas.pixel_scale
        w_log, h_log = self.canvas.size
        tex_w, tex_h = int(w_log * ps), int(h_log * ps)

        self.color_texture = gloo.Texture2D(shape=(tex_h, tex_w, 4), format='rgba')
        self.fbo = gloo.FrameBuffer(color=self.color_texture, depth=None)
        self.voxel = None
        self.selected_index = None
        self.selected_vector = None
        self.box_start = None
        self.box_end = None
        self._global_box = [None] * 5
        # actual roi
        self._actual_roi_global_by_level = [None] * 5

    def set_actual_roi_global(self, lvl: int, gs, ge):
        gs = np.asarray(gs, dtype=float)
        ge = np.asarray(ge, dtype=float)
        self._actual_roi_global_by_level[lvl] = (gs, ge)


    def handle_click(self, x, y, w, h, pos_visuals, shapes, transforms):
        if self.color_texture.shape[0] != h or self.color_texture.shape[1] != w:
            self.color_texture = gloo.Texture2D(shape=(h, w, 4), format='rgba')
            self.fbo = gloo.FrameBuffer(color=self.color_texture, depth=None)
        best_index = -1
        best_val = -1
        best_vector = None
        print("[VolumeSelector] pos_visuals len", len(pos_visuals))
        # find volume
        for i, pos_vis in enumerate(pos_visuals):
            print(i)
            pos_vis.visible = True
            with self.fbo:
                gloo.clear()
                pos_vis.draw()
                img = gloo.read_pixels((0, 0, w, h),
                                       mode='color', out_type='float')
                val = np.sum(img[y][x][:3])

                if val > best_val:
                    best_val = val
                    best_index = i
                    print("[VolumeSelector] best_index: ",best_index)
                    best_vector = np.array(img[y][x][:3]) * shapes[i]
                gloo.clear()
            pos_vis.visible = False

        if best_index != -1:
            self.selected_index = best_index
            self.selected_vector = best_vector

            return best_index, best_vector

        return None, None

    def draw_box(self, vector, translation):
        # Remove previous box if any
        if self.voxel:
            self.voxel.parent = None

        # Compute box size
        depth, height, width = self.shape  # Z, Y, X
        box_size = np.array([width // 2, height // 2, depth // 2])
        # box_size = np.array([width // 2, depth // 2, height // 2])
        color = (0., 0., 0., 0.01)

        # Convert local vector (X,Y,Z) + world translation → world position (X,Y,Z)
        print("[VolumeSelector] translation:", translation)
        world_pos = np.array([vector[2], vector[1], vector[0]]) + translation[::-1]
        print("[VolumeSelector] box_size:", box_size)

        # Record ROI start/end in (Z,Y,X)
        self.box_start = (world_pos - box_size / 2)[::-1]
        self.box_end = (world_pos + box_size / 2)[::-1]
        print("[VolumeSelector] box start:", self.box_start, "end:", self.box_end)

        # Size to draw
        plot_box_size = self.box_start - self.box_end
        print(plot_box_size)

        # Create the 3D box
        self.voxel = Box(plot_box_size[2], plot_box_size[0], plot_box_size[1],
                         color=color, edge_color='Yellow')
        self.voxel.set_gl_state(blend=True, blend_equation='max',
                                blend_func=('one', 'one'),
                                line_width=6.0, depth_test=False)
        self.voxel.parent = self.view.scene
        self.voxel.transform = STTransform(translate=tuple(world_pos))

    def _update_box_corners(self, center, half_extents=None):
        if half_extents is None:
            d, h, w = self.shape
            half_extents = np.array([w//2, h//2, d//2], dtype=float)

        if center is None:
            center = np.array(self.voxel.transform.translate, dtype=float)

        self.box_start = center[:3] - half_extents/2
        self.box_end   = center[:3] + half_extents/2

    def move_box(self, delta_xyz):
        if self.voxel is None:
            return

        # 1) Get current world_pos (X, Y, Z) from the existing transform
        current_pos = np.array(self.voxel.transform.translate, dtype=float)

        # 2) New world_pos = current + delta
        new_pos = current_pos[:3] + delta_xyz
        print("[VolumeSelector] move_box: current_pos =", current_pos,
              "delta =", delta_xyz, "-> new_pos =", new_pos)

        # 3) Recompute box_start / box_end in global (Z, Y, X) at the current LOD layer
        depth, height, width = self.shape  # (Z, Y, X)
        box_size = np.array([width // 2, height // 2, depth // 2])  # (X, Y, Z)
        # box_size = np.array([width // 2, depth // 2, height // 2])  # (X, Y, Z)
        self.box_start = (new_pos - box_size / 2)[::-1]
        self.box_end = (new_pos + box_size / 2)[::-1]
        print("[VolumeSelector] move_box -> updated box_start (Z,Y,X) =", self.box_start,
              ", box_end (Z,Y,X) =", self.box_end)

        plot_box_size = self.box_start - self.box_end
        print(plot_box_size)

        # Remove previous box, if any, and create a new one with updated size/position
        color = (0., 0., 0., 0.01)
        if self.voxel:
            self.voxel.parent = None
        self.voxel = Box(plot_box_size[2], plot_box_size[0], plot_box_size[1],
                         color=color, edge_color='Yellow')
        # self.voxel = Box(width // 2, depth // 2, height // 2, color=color, edge_color='Yellow')
        self.voxel.set_gl_state(blend=True, blend_equation='max',
                                blend_func=('one', 'one'),
                                line_width=6.0, depth_test=False)
        self.voxel.parent = self.view.scene
        self.voxel.transform = STTransform(translate=tuple(new_pos))

    def refresh_box(self):

        if self.voxel is None:
            return

        self._update_box_corners(center=None)

    def remove_box(self):
        if self.voxel:
            self.voxel.parent = None
            self.voxel = None

    # def draw_box_in_global(self, gview, f, lvl):
    #     max_f = f[0]
    #     f = f[lvl]
    #
    #     # Remove existing global boxes
    #     for _global_box in self._global_box:
    #         if _global_box:
    #             _global_box.parent = None
    #
    #     # 1) Convert box_start/end back to original-resolution global voxel coordinates
    #     gs = np.array(self.box_start) * f
    #     ge = np.array(self.box_end) * f
    #
    #     print(f"[VolumeSelector] box_start (pre original-res scaling): {self.box_start}")
    #     print(f"[VolumeSelector] box_end   (pre original-res scaling): {self.box_end}")
    #     print(f"[VolumeSelector] factor f: {f}")
    #     print(f"[VolumeSelector] gs (global start, original-res): {gs}")
    #     print(f"[VolumeSelector] ge (global end,   original-res): {ge}")
    #
    #     # 2) Compute center and half-extent (in original resolution)
    #     center = (gs + ge) / 2.0
    #     half = (ge - gs) / 2.0
    #
    #     print(f"[VolumeSelector] center (global center [z, y, x]): {center}")
    #     print(f"[VolumeSelector] half   (half-extent [z/2, y/2, x/2]): {half}")
    #
    #     # 3) Create Box visual
    #     size = (np.array(half) * 2 / max_f)[::-1]
    #     print(f"[VolumeSelector] box size (width, height, depth) = {size}")
    #     color = (0., 0., 0., 1.0)
    #     box = Box(size[0], size[2], size[1],
    #               color=color,
    #               edge_color='yellow')
    #
    #     # 4) Set GL state and parent
    #     box.set_gl_state(blend=True, blend_equation='max',
    #                      blend_func=('one', 'one'),
    #                      line_width=6.0, depth_test=False)
    #     box.parent = gview.scene
    #
    #     # 5) Translate to global position (scaled to the overview canvas space)
    #     tx, ty, tz = center[::-1] / max_f
    #     print(f"[VolumeSelector] translate = (tx, ty, tz) = ({tx}, {ty}, {tz})")
    #     box.transform = STTransform(translate=(tx, ty, tz))
    #
    #     self._global_box[lvl] = box
    def draw_box_in_global(self, gview, f, globox_in_ldr, lvl):
        """
        globox_in_ldr: list of [start, end] for each loader,
                       start/end are global original-res coords in (Z, Y, X).
                       例如: [[gs0, ge0], [gs1, ge1], ...]
        f: factor list (傳 loaders[0].factor)
        """
        max_f = f[0]

        # 0) 移除舊的 global boxes
        # for i, _global_box in enumerate(self._global_box):
        #     if _global_box:
        #         _global_box.parent = None
        #         self._global_box[i] = None
        for _global_box in self._global_box:
            if _global_box:
                _global_box.parent = None

        if not globox_in_ldr or len(globox_in_ldr) == 0:
            print("[VolumeSelector] draw_box_in_global: no ROIs provided; skip.")
            return

        cleaned = []
        for i, pair in enumerate(globox_in_ldr):
            if pair is None or len(pair) != 2:
                continue
            gs = np.asarray(pair[0], dtype=float)
            ge = np.asarray(pair[1], dtype=float)

            if gs.shape[-1] >= 4:
                gs = gs[:3]
            if ge.shape[-1] >= 4:
                ge = ge[:3]

            if np.any(ge <= gs):
                print(f"[VolumeSelector] skip invalid ROI[{i}] start={gs} end={ge}")
                continue

            cleaned.append((gs, ge))
            print(f"[SelectionBox] ROI[{i}] start={gs.astype(int)} end={ge.astype(int)} "
                  f"size={(ge - gs).astype(int)}")

        if not cleaned:
            print("[VolumeSelector] draw_box_in_global: no valid ROIs after cleaning; skip.")
            return

        union_gs = cleaned[0][0].copy()
        union_ge = cleaned[0][1].copy()
        for gs, ge in cleaned[1:]:
            union_gs = np.minimum(union_gs, gs)
            union_ge = np.maximum(union_ge, ge)

        union_size_zyx = union_ge - union_gs
        if np.any(union_size_zyx <= 0):
            print(f"[VolumeSelector] union ROI is empty: start={union_gs}, end={union_ge}; skip.")
            return

        print(f"[SelectionBox] UNION start={union_gs.astype(int)} end={union_ge.astype(int)} "
              f"size={(union_size_zyx).astype(int)}")

        center_zyx = (union_gs + union_ge) / 2.0
        size_overview_xyz = (union_size_zyx / max_f)[::-1]  # (X, Y, Z)
        tx, ty, tz = (center_zyx[::-1] / max_f)  # (X, Y, Z)

        color = (0., 0., 0., 1.0)
        box = Box(size_overview_xyz[0], size_overview_xyz[2], size_overview_xyz[1],
                  color=color, edge_color='yellow')
        box.set_gl_state(blend=True, blend_equation='max',
                         blend_func=('one', 'one'),
                         line_width=6.0, depth_test=False)
        box.parent = gview.scene
        box.transform = STTransform(translate=(tx, ty, tz))

        self._global_box[lvl] = box

    def _removeGlobalBox(self, lvl):
        if self._global_box[lvl] is not None:
            self._global_box[lvl].parent = None
            self._global_box[lvl] = None

    def _reload_global(self, view, lvl):
        if lvl >= 0:
            print('[VolumeSelector] reload glo',lvl, self._global_box[lvl])
            # if self._global_box[lvl].parent == None:
            self._global_box[lvl].parent = view.scene

