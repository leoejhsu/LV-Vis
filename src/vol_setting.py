# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh


# Volume Data Structure (aligned with vol_setting.py)
#
# ├─ MultiVolumeController                          # Controller of multiple SingleVolumeLOD
# │  ├─ view: SceneCanvas                           # Interactive (main) canvas
# │  ├─ gView: SceneCanvas                          # Global/overview canvas
# │  ├─ volumes: dict[int, SingleVolumeLOD]         # {vid → loader}
# │  ├─ current_layer: int                          # Current LOD level (0 = coarsest)
# │  ├─ shape: np.ndarray | None                    # vol_size (Z,Y,X) of the first loaded volume
# │  ├─ next_id: int                                # Auto-increment volume id
# │  ├─ folder: list[str]                           # (reserved; not used)
# │  └─ add_volume(csv_path: str) -> list[int]      # Load volumes from a 4-column CSV
# │     ├─ CSV row: [path, x, y, z] (no header)
# │     ├─ raw_pos = (z, y, x)                      # Stored internally as (Z,Y,X)
# │     ├─ loader = SingleVolumeLOD(vid, folder, raw_pos, img_name=basename(folder))
# │     ├─ loader.load_level0(folder)               # Detect max_level; load `{img_name}.npy`; normalize to [0,1]
# │     ├─ loader.render_level(0, view)             # Render into interactive view
# │     └─ loader.render_level(0, gView, globalV=True)  # Render into global view
# └─ SingleVolumeLOD                                # One volume: I/O, extraction, rendering
#    ├─ id: int                                     # Unique id assigned by controller
#    ├─ folder: str                                 # Folder containing `{img_name}.npy` and LOD subfolders
#    ├─ img_name: str
#    ├─ layer: int                                  # Current LOD level
#    ├─ vol_size: np.ndarray[int]                   # Shape (Z, Y, X)
#    ├─ raw_positions: np.ndarray[3]                # World-space start (Z, Y, X)
#    ├─ translations: list[int]                     # Extra offset [dz, dy, dx] in voxels
#    ├─ max_level: int                              # Inferred by scanning folder depth
#    ├─ factor: list[int]                           # Length = max_level+1, e.g., [2^L, …, 1]
#    ├─ vol_global_start_point: list[np.ndarray]    # Per-level global start used during extraction
#    ├─ volumes: list[np.ndarray | None]            # Per-level data arrays, indices 0..max_level
#    ├─ local_positions: list | None                # Placeholder; not used in current code
#    ├─ layer_dict: list[dict[str, np.ndarray]]     # Per-level sub-block global starts (via layer_n_dict)
#    ├─ layer_dict_corners: list[dict[str, np.ndarray]]  # 8-corner coords per block (via dict_corners)
#    ├─ volume_visuals: Visual | None               # Interactive view visual (MyVisual) for current level
#    ├─ gloVolume_visuals: Visual | None            # Global/overview visual (MyVisual)
#    ├─ volume_pos_visuals: Visual | None           # Hidden picking visual (MyVolPosVisual)
#    ├─ load_level0(path: str, level: int = 0)      # Set max_level, compute factors, load/normalize npy, init octree, store L0
#    ├─ init_octree(vol_size: np.ndarray)           # Build layer_dict / layer_dict_corners from max_level
#    ├─ render_level(level: int, view, threshold: float = 0.25, globalV: bool = False)
#    └─ extract_next(center: np.ndarray, lvl: int) -> np.ndarray   # Pull next-level subvolume and update starts

# ────────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
from vispy import scene
from vispy.visuals.transforms import STTransform
from .utils.volume_extract import extract_volume_from_next_layer
from .my_vispy.my_volume import MyVisual
from .my_vispy.my_volume_pos import MyVolPosVisual
from .utils.volume_dict import dict_corners, layer_n_dict
import gc
from vispy.gloo import gl

import pandas as pd


def _try_delete(obj, tag=""):
    try:
        if obj is not None and hasattr(obj, "delete"):
            obj.delete()
            # print(f"[dispose] deleted {tag}")
    except Exception as e:
        print(f"[dispose] delete {tag} error:", e)


def get_max_lod_level(root_dir):
    max_level = 0
    for root, dirs, files in os.walk(root_dir):
        relative_path = os.path.relpath(root, root_dir)
        if relative_path == ".":
            level = 0
        else:
            level = len(relative_path.split(os.sep))
        max_level = max(max_level, level)
    return max_level

class SingleVolumeLOD:
    def __init__(self,
                 vid: int,
                 folder: str,
                 # view: scene.SceneCanvas,
                 # globalView: scene.SceneCanvas,
                 raw_positions: np.ndarray,
                 img_name: str):
        # Core attributes
        self.id       = vid             # Unique identifier for this volume
        self.folder   = folder          # Absolute folder path
        self.img_name = img_name        # Subfolder/base name for the .npy data


        # Dynamic state
        self.layer    = 0               # Current LOD level (0 = coarsest)
        self.vol_size = None            # Volume shape as np.ndarray([Z, Y, X])

        # Per-volume spatial data
        self.raw_positions = raw_positions  # World-space start (Z, Y, X)
        self.translations  = [0, 0, 0]      # Extra offset [dz, dy, dx] in voxels

        # Visual handles
        self.volume_visuals     = None  # MyVisual in interactive view
        self.gloVolume_visuals  = None  # MyVisual in global/overview view
        self.volume_pos_visuals = None  # MyVolPosVisual for picking (hidden)

        # Drill-down / octree state

        # Factor & dictionaries (placeholders; finalized in load_level0 / init_octree)
        self.factor                   = [16, 8, 4, 2, 1]  # Will be recomputed in load_level0()
        self.max_level                = 0                 # Inferred from directory depth
        self.layer_dict               = None             # Per-level sub-block global starts
        self.layer_dict_corners       = None             # Per-level 8-corner coordinates
        self.vol_global_start_point   = None             # List[np.ndarray] per level (global starts)
        self.volumes                  = None             # List[np.ndarray | None] per level
        self.local_positions          = None             # Reserved for per-level local offsets

        self.vol_global_end_point = None  # List[np.ndarray] per level (global ends)
        self.data_min = None
        self.data_max = None


    def load_level0(self, path, level=0):
        self.max_level = get_max_lod_level(path)
        print('[SingleVolumeLOD] max level', self.max_level)

        self.vol_global_start_point = [np.zeros(3, dtype=int) for _ in range(self.max_level + 1)]
        self.vol_global_end_point = [np.zeros(3, dtype=int) for _ in range(self.max_level + 1)]
        self.volumes              = [None] * (self.max_level+1)
        self.local_positions      = [None] * (self.max_level+1)
        self.factor = [2**(self.max_level-i) for i in range(self.max_level+1)]

        print('[SingleVolumeLOD] factors: ',self.factor)
        path = os.path.join(path, f"{self.img_name}.npy")

        npy_path = path
        # load and normalizing
        data = np.load(npy_path).astype(np.float32)
        self.data_min = float(data.min())
        self.data_max = float(data.max())
        print('[SingleVolumeLOD] factors: ', self.data_max, self.data_min)
        # data = (data - data.min())/(data.max()-data.min())


        self.vol_size = np.array(data.shape, dtype=int)
        self.init_octree(self.vol_size)
        # print(f'[SingleVolumeLOD] layer_dict', self.layer_dict)
        # print(f'[SingleVolumeLOD] layer_dict_corners', self.layer_dict_corners[0])
        self.volumes[level]       = data

    def init_octree(self, vol_size: np.ndarray):
        max_level = self.max_level

        self.layer_dict = [
            layer_n_dict(vol_size, level+1, max_level)
            for level in range(max_level)
        ]

        self.layer_dict_corners = [
            dict_corners(layer_dict, vol_size, factor=2 ** (max_level - (level+1)))
            for level, layer_dict in enumerate(self.layer_dict)
        ]

    def render_level(self, level:int, view:scene.SceneCanvas, threshold:float=0.25, globalV:bool=False):
        if not globalV:
            if self.volume_visuals:
                self.volume_visuals.parent = None
            if self.volume_pos_visuals:
                self.volume_pos_visuals.parent = None

        # 2) calculate positions

        data = self.volumes[level]
        assert data is not None, "[render_level] data is None"
        print(
            f"[render_level] level={level} data.shape={getattr(data, 'shape', None)} dtype={getattr(data, 'dtype', None)}")

        raw  = self.raw_positions + self.translations + self.vol_global_start_point[level]
        f    = self.factor[level]

        # local = ((raw + self.vol_global_start_point[level]) // f).astype(int)
        local = (raw // f).astype(int)
        print(f"[SingleVolumeLOD] level {level}, raw pos: {raw}, local: {local}")
        translate = local[::-1]

        # rendering

        if data.ndim == 3:
            tex_fmt = 'auto'
        elif data.ndim == 4 and data.shape[-1] in (2, 3, 4):
            tex_fmt = 'rgba'
        else:
            raise ValueError(f"[render_level] Unexpected data shape: {data.shape}")

        MyNode = scene.visuals.create_visual_node(MyVisual)
        vis = MyNode(data, parent=view.scene, threshold=threshold)
        vis.transform = STTransform(translate=tuple(translate))
        vis.set_gl_state(
            blend=True,
            blend_equation='max',
            blend_func=('one', 'one')
        )
        vis.shared_program['u_min_val'] = 0.0
        vis.shared_program['u_max_val'] = 1.0
        if globalV:
            self.gloVolume_visuals = vis
        else:
            self.volume_visuals = vis

        # position canvas rendering
        PosNode = scene.visuals.create_visual_node(MyVolPosVisual)
        pvis = PosNode(data, parent=view.scene, threshold=threshold)
        pvis.transform = STTransform(translate=tuple(translate))
        pvis.set_gl_state(
            blend=True,
            blend_equation='max',
            blend_func=('one', 'one')
        )
        pvis.visible = False
        if not globalV:
            self.volume_pos_visuals = pvis

    def extract_next(self, center:np.ndarray, lvl:int):
        print("[SingleVolumeLOD] center: ", center)
        out, new_starts, new_ends = extract_volume_from_next_layer(
            self, center, self.vol_global_start_point, self.vol_global_end_point, self.folder, lvl
        )
        self.volumes[lvl+1] = out
        self.vol_global_start_point = new_starts
        self.vol_global_end_point = new_ends
        print("[SingleVolumeLOD] global point: ", self.vol_global_start_point, self.vol_global_end_point)

        return out

    def dispose_visual(self, level: int, globalV: bool = False, pos: bool = False):
        vis = None
        if not globalV and not pos:
            vis, self.volume_visuals = self.volume_visuals, None
        elif globalV and not pos:
            vis, self.gloVolume_visuals = self.gloVolume_visuals, None
        elif not globalV and pos:
            vis, self.volume_pos_visuals = self.volume_pos_visuals, None

        if vis is None:
            return

        if hasattr(vis, "dispose"):
            try:
                vis.dispose()
            except Exception as e:
                print("[dispose_visual] vis.dispose() error:", e)

        for attr, tag in (
            ("_texture", "texture"),
            ("_tex", "tex"),
            ("_tex3d", "tex3d"),
            ("_vbo", "vbo"),
            ("_ibo", "ibo"),
            ("_fbo", "fbo"),
            ("_depth_buffer", "depth_buffer"),
            ("_color_buffer", "color_buffer"),
            ("_program", "program"),
            ("shared_program", "shared_program"),
        ):
            obj = getattr(vis, attr, None)
            _try_delete(obj, tag)

        try:
            vis.parent = None
        except Exception:
            pass

        try:
            gl.glFlush()
            gl.glFinish()
        except Exception:
            pass

    def free_level(self, level: int):

        self.dispose_visual(level, globalV=False, pos=False)
        # self.dispose_visual(level, globalV=True,  pos=False)
        self.dispose_visual(level, globalV=False, pos=True)

        if level < len(self.volumes) and self.volumes[level] is not None:
            self.volumes[level] = None
            self.vol_global_start_point[level] = [0, 0, 0]
            self.vol_global_end_point[level] = [0, 0, 0]

        gc.collect()

    def has_level(self, level: int) -> bool:
        return (level < len(self.volumes) and self.volumes[level] is not None)


    def has_visual(self, level: int) -> bool:
        return level in getattr(self, 'visuals_by_level', {})

    def activate_level(self, level: int, view):
        vis = self.visuals_by_level[level]
        if vis.parent is None:
            view.add(vis)



class MultiVolumeController:
    """
    Multiple SingleVolumeLOD controller
    """
    def __init__(self,
                 view: scene.SceneCanvas,
                 gView: scene.SceneCanvas,
                 # factors: list[int],
                 ):

        self.view      = view
        self.gView     = gView
        # self.factors   = factors
        self.folder    = []
        # self.n_levels  = len(factors)

        # volume
        self.volumes       = {}    # vid -> SingleVolumeLOD
        self.current_layer = 0
        self.shape         = None
        self.next_id       = 0

    def add_volume(self, csv_path: str) -> list[int]:
        """
        Load volumes from a CSV (no header, exactly 4 columns: [path, x, y, z]).
        Returns a list of assigned volume IDs.
        """
        # Reset controller state
        self.volumes.clear()
        self.next_id = 0
        self.current_layer = 0

        # Strict CSV: no header, exactly 4 columns
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] != 4:
            raise ValueError("Invalid CSV format: expected four columns [path, x, y, z] with no header.")

        new_ids = []
        for idx, row in df.iterrows():
            folder = str(row[0]).strip()
            if not folder:
                raise FileNotFoundError(f"Row {idx + 1}: empty path in CSV")

            if not os.path.isabs(folder):
                folder = os.path.join(os.getcwd(), folder)

            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Row {idx + 1}: resolved path does not exist or is invalid: {folder!r}"
                )

            try:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except Exception:
                raise ValueError(
                    f"Row {idx + 1}: coordinates must be numeric: x={row[1]!r}, y={row[2]!r}, z={row[3]!r}")

            # Internal storage uses (Z, Y, X)
            raw_pos = np.array([z, y, x], dtype=np.float32)

            vid = self.next_id
            loader = SingleVolumeLOD(
                vid=vid,
                folder=folder,
                raw_positions=raw_pos,
                img_name=os.path.basename(folder)
            )

            loader.load_level0(folder)
            loader.render_level(0, self.view)
            loader.render_level(0, self.gView, globalV=True)

            if vid == 0:
                # Record the reference shape from the first volume
                self.shape = loader.vol_size

            self.volumes[vid] = loader
            new_ids.append(vid)
            self.next_id += 1

        return new_ids

