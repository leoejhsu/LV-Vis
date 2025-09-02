"""
LOD Preprocessing Pipeline (CLI)
Transforms a 3D volume (TIFF series folder or a single 3D TIFF stack) into an
octree/LOD directory of .npy blocks that your viewer can stream efficiently.

What this script does
- Detects input mode automatically:
  1) CSV file (one path per line; no header) — each row is either a folder of TIFFs
     or a single multi-page TIFF stack.
  2) Folder of 2D TIFF slices (TIFF series).
  3) Single 3D TIFF stack (.tif/.tiff).
- Estimates a maximum LOD depth by repeatedly halving the mean(Z,Y,X) until < 100.
- Creates an octree-like folder tree under `<input>/LOD_Data` (configurable via `lod_name`).
- Crops the original stack into leaf blocks (zero-padded to align the grid), then
  assembles parent nodes bottom-up using 2×2×2 max-pooling (`block_reduce`).
- Writes `.npy` chunks per node and a `LOD_log.txt` with timings and notes.

Key functions
- `oneVolProcessing(file_path, dtype=None, stack_path=None, ..., lod_name="LOD_Data")`:
   Process one volume (folder or stack) into the LOD directory.
- `compute_maximum_level(shape, min_size=100)`:
   Return number of halvings used to define the LOD depth.
- `make_folder_tree(...)`, `crop_original_data(...)`, `assemble_level(...)`:
   Build the directory layout, write leaf nodes, then assemble upper levels.

Input formats
- CSV (for this *preprocessor*): **one column only**, no header; each row is a path
  to either a TIFF series folder or a single multi-page TIFF file.
  (Note: This differs from the *viewer* CSV, which uses 4 columns: `path,x,y,z`.)
- Folder: contains ordered TIFF slices (the script sorts by the last integer in the filename).
- Single 3D TIFF: a multi-page `.tif/.tiff`.

Outputs
- `<input_folder>/<lod_name>/...` octree of `.npy` blocks (leaf to root).
- `<input_folder>/LOD_log.txt` with progress messages and timing.

How to run (examples)
    # Folder of TIFF slices, default dtype=uint16
    python data_preprocessing_adpated.py /data/rod1/part3_1

    # Single 3D TIFF stack, force float32
    python data_preprocessing_adpated.py /data/rod1/stack.tiff --dtype float32

    # CSV (one path per line; each line: folder OR single .tiff)
    python data_preprocessing_adpated.py ./paths_list.csv --dtype uint16

Assumptions
- TIFFs are grayscale; dtype can be inferred from the first slice if not provided.
- Sufficient disk space for intermediate `.npy` blocks.
- If integrating with a GUI, `progress_cb / stage_cb / abort_cb` can be passed through.

Notes
- This CLI version only changes the *dtype* and input path; the LOD folder name can
  be overridden from the GUI by passing `lod_name` into `oneVolProcessing`.
- Padding ensures each leaf block grid aligns exactly at the chosen LOD depth.
"""

import os
import numpy as np
from tqdm import tqdm, trange
from skimage.measure import block_reduce
import tifffile as tiff
import re
import time
import pandas as pd
import gc
from itertools import product
import argparse

class LazyTiffSlices:
    """Lazy reader for a multi-page TIFF stack.
    Parameters
    ----------
    tiff_file : tifffile.TiffFile
    An open `tifffile.TiffFile` instance.
    dtype : np.dtype
    Desired dtype for returned slices (e.g., np.uint16).
    """
    def __init__(self, tiff_file, dtype):
        self.tif = tiff_file
        self.dtype = dtype
        self.length = len(tiff_file.pages)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.tif.pages[idx].asarray().astype(self.dtype)

def binary_corners(dim=3):
    """Return all 2^dim binary corner coordinates as int array of shape (2^dim, dim)."""
    return np.array(list(product([0, 1], repeat=dim)))

# folder tree
def make_folder_tree(directory, level, max_level):
    """Create an octree-like folder structure recursively.
    Example for max_level=2:
    directory/
    0_0/ 0_1/ … 0_7/
    1_0/ … 1_7/
    2_0/ … 2_7/

    Parameters
    ----------
    directory : str
    Root output directory.
    level : int
    Current level to create (call with 0 initially).
    max_level : int
    Deepest level to create.
    """
    if level > max_level:
        return
    for i in range(8):
        new_dir = os.path.join(directory, f"{level}_{i}")
        os.makedirs(new_dir, exist_ok=True)
        make_folder_tree(new_dir, level + 1, max_level)

# # crop original data and allocate in folder
def crop_original_data(item_list, image_shape, file_path, dtype, log, max_level, progress_cb=None, stage_cb=None, abort_cb=None):
    """Partition the original stack into lowest-level child blocks and save them.

    High-level idea
    ---------------
    - We iterate over groups of Z slices and pack them into a zero-padded 3D
    buffer whose (Z, Y, X) are multiples of 2^L, so that later (Y, X) tiles
    align with the octree grid.
    - Each buffer region is saved to the leaf folders according to its child
    index path (e.g., `0_a/1_b/…/L_i.npy`).

    Parameters
    ----------
    item_list : sequence of path|array|callable
    Each element is either (a) a slice path, (b) a 2D ndarray, or (c) a
    no-arg function returning a 2D ndarray (lazy loader). The sequence
    length is Z.
    image_shape : tuple(int,int,int)
    Original 3D shape as (Z, Y, X).
    out_root : str
    Root folder to store the octree (`make_folder_tree` should be called
    beforehand).
    dtype : np.dtype
    Output dtype for `.npy` chunks.
    log : file-like
    Open file handle for writing progress messages.
    max_level : int
    Depth of the leaf level (coarsest is level 0). Leaf nodes live at
    `level = max_level`.
    progress_cb, stage_cb, abort_cb : optional callbacks
    UI hooks; safe to leave as `None` in CLI usage.
    """
    max_level2time = 2 ** max_level
    start_time = time.time()
    z_stack_per_op = np.ceil(len(item_list) / max_level2time).astype(np.int64)
    y_shape = (np.ceil(image_shape[1] / max_level2time) * max_level2time).astype(np.int64)
    x_shape = (np.ceil(image_shape[2] / max_level2time) * max_level2time).astype(np.int64)
    images = np.zeros((z_stack_per_op, y_shape, x_shape), dtype=dtype)
    image_downscale_shape = np.ceil(np.array(images.shape) / max_level2time).astype(np.int64)
    down_y, down_x = image_downscale_shape[1], image_downscale_shape[2]
    xyz_list = binary_corners(max_level)
    z_stack = 0
    for k in range(max_level2time):

        if abort_cb and abort_cb(): return
        if stage_cb: stage_cb(f"Cropping batch {k + 1}/{max_level2time}")

        xyz_z_update_list = xyz_list + xyz_list[k] * 4
        images.fill(0)
        for q in range(z_stack_per_op):
            try:
                slice_data = item_list[k * z_stack_per_op + q]
                if isinstance(slice_data, np.ndarray):
                    image = slice_data
                elif isinstance(slice_data, str):
                    image = tiff.imread(slice_data).astype(dtype)
                else:  # lazy loader like LazyTiffSlices
                    # image = slice_data.__getitem__(k * z_stack_per_op + q)
                    image = slice_data()

                if k == 0:
                    print(f"[Debug] Slice {q} max={image.max()}, min={image.min()}, shape={image.shape}")

                images[q, :image.shape[0], :image.shape[1]] = image
                del image

            # except Exception as e:
            #
            #     print(f"[Error @ k={k}, q={q}]: {e}")
            except:
                pass


        for j in range(max_level2time):
            xyzj_update_list = xyz_z_update_list + xyz_list[j] * 2
            for i, xyz in enumerate(xyzj_update_list):
                arr = images[:, j * down_y: j * down_y + down_y, i * down_x: i * down_x + down_x]
                save_path = os.path.join(*[f"{i}_{v}" for i, v in enumerate(xyz)])
                filename = f"{len(xyz) - 1}_{xyz[-1]}"
                print(save_path)
                print(arr.shape)
                np.save(os.path.join(file_path, f"{save_path}/{filename}.npy"), arr.astype(dtype))
        z_stack += z_stack_per_op
        gc.collect()
        if progress_cb:
            progress_cb(5 + int(55 * (k + 1) / max_level2time))

    log.write(f"crop_original_data done in {time.time() - start_time:.2f} seconds\n")

# assemble data and downscale
def assemble_and_downscale(dir, dtype, log):
    """Assemble 8 child blocks in `dir_path` and write a parent block (.npy).
    Expects files named `<basename>.npy` inside each child folder, where
    `<basename>` equals the child folder name (e.g., `…/1_3/1_3.npy`).
    After assembling a (2×,2×,2×) cube of children, the function applies
    `block_reduce(..., (2,2,2), np.max)` and writes `<dir_basename>.npy` into
    the parent directory.
    """
    start_time = time.time()
    items = os.listdir(dir)
    item_list = sorted([item for item in items if (".npy" not in item and ".txt" not in item)])
    xyz_list = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    item0_path = os.path.join(dir, f"{item_list[0]}/{item_list[0]}.npy")
    original_shape = np.array(np.load(item0_path).shape)
    arr = np.zeros(original_shape * 2, dtype=dtype)
    for i, item in enumerate(item_list):
        arr_part = np.load(os.path.join(dir, f"{item}/{item}.npy")).astype(dtype)
        load_shape = xyz_list[i] * original_shape
        arr[load_shape[0]:load_shape[0]+original_shape[0], load_shape[1]:load_shape[1]+original_shape[1], load_shape[2]:load_shape[2]+original_shape[2]] = arr_part
    arr = block_reduce(arr, block_size=(2, 2, 2), func=np.max)
    save_path = os.path.join(dir, dir.split("/")[-1])
    np.save(f"{save_path}.npy", arr.astype(dtype))
    # with open(f"{save_path}.txt", "w") as file:
    #     file.write(f"{data_coordinate}\n")
    log.write(f"assemble_and_downscale {dir} done in {time.time() - start_time:.2f} seconds\n")

def assemble_level(folder_path, dtype, pad_image_shape, log, max_level,
                   progress_cb=None, stage_cb=None, abort_cb=None):
    """Walk from the deepest level up to the root and build parent nodes.

    Parameters
    ----------
    folder_path : str
    Root of the octree directory created by `make_folder_tree`.
    dtype : np.dtype
    Output dtype for saved `.npy` volumes.
    log : file-like
    Open file handle for writing progress messages.
    max_level : int
    Deepest level to assemble (leaf level). The function assembles all
    parents up to the root.
    progress_cb, stage_cb, abort_cb : optional callbacks
    UI hooks; safe to omit in CLI.
    """
    total = sum(8 ** lvl for lvl in range(max_level, 0, -1))
    done = 0

    for level in range(max_level, 0, -1):
        if stage_cb:
            stage_cb(f"Assembling level {level}…")

        next_level = level - 1
        for idx in product(range(8), repeat=next_level):
            if abort_cb and abort_cb():
                log.write("assemble_level aborted by user\n")
                return

            for i in range(8):
                sub_idx = list(idx) + [i]
                sub_path = os.path.join(folder_path, *[f"{l}_{v}" for l, v in enumerate(sub_idx)])
                print(sub_path)

                assemble_and_downscale(sub_path, dtype, log)

                done += 1
                if progress_cb:
                    progress_cb(60 + int(35 * done / total))

    if stage_cb:
        stage_cb("Assembling root…")

    assemble_and_downscale(folder_path, dtype, log)

    if progress_cb:
        progress_cb(95)



def compute_maximum_level(image_3d_shape, min_size=100):
    """Estimate how many times the average dimension can be halved until < min_size.
    Example
    -------
    If the mean of (Z, Y, X) is 800 and `min_size` is 100, then the levels are:
    800 → 400 → 200 → 100 (stops before going below 100) ⇒ 3 levels.

    Returns the number of *halvings* (an integer ≥ 0).
    """
    z, y, x = image_3d_shape
    base_size = np.mean([z, y, x])

    level = 0
    while base_size >= min_size * 2:
        base_size //= 2
        level += 1
    return level

def oneVolProcessing(file_path, dtype=None, stack_path=None,
                     progress_cb=None, stage_cb=None, abort_cb=None,
                     lod_name="LOD_Data", min_size=100):   # 👈 增加可傳入的 LOD 資料夾名稱
    """Process one volume folder or one stack into an LOD directory.

    Parameters
    ----------
    file_path : str
    If `stack_path` is given, this is the folder where the output directory
    will be created. Otherwise, this is the folder containing a TIFF series.
    dtype : np.dtype, optional
    Output dtype. If omitted, inferred from the first slice.
    stack_path : str, optional
    Path to a single multi-page TIFF (3D stack). If provided, `file_path`
    should be its parent folder.
    lod_name : str
    Name of the output LOD directory.
    min_size : int
    Target minimal size used to estimate the maximum LOD depth.
    """
    log_path = os.path.join(file_path, "LOD_log.txt")
    with open(log_path, "w") as log:
        # loading data
        if stack_path is not None:
            tif = tiff.TiffFile(stack_path)
            lazy_loader = LazyTiffSlices(tif, np.uint16 if dtype is None else dtype)

            image0 = lazy_loader[0]
            if dtype is None:
                dtype = image0.dtype
            image_3d_shape = (len(lazy_loader), image0.shape[0], image0.shape[1])
            item_list = [lambda idx=i: lazy_loader[idx] for i in range(len(lazy_loader))]
        else:
            file_dir = file_path
            items = [i for i in os.listdir(file_dir) if '.tif' in i]
            items = sorted(items, key=lambda x: int(re.findall(r'\d+', x)[-1]))
            image0 = tiff.imread(os.path.join(file_dir, items[0]))
            if dtype is None:
                dtype = image0.dtype
            image0 = image0.astype(dtype)
            item_list = [os.path.join(file_dir, item) for item in items]
            image_3d_shape = (len(item_list), image0.shape[0], image0.shape[1])

        print('calculated shape: ', image_3d_shape)
        print('using dtype:', dtype)

        # calculate LOD layer ===
        maximum_level = compute_maximum_level(image_3d_shape, min_size=min_size)
        print('maximum_level: ', maximum_level)

        pad_image_shape = np.ceil(np.array(image_3d_shape) / (2 ** maximum_level)).astype(np.int64)

        # generate file structure
        lod_path = os.path.join(file_path, lod_name)
        os.makedirs(lod_path, exist_ok=True)
        level = 0
        make_folder_tree(lod_path, level, maximum_level - 1)

        log.write(f"Processing started with padded shape: {pad_image_shape}\n")

        # leaf layer calculation
        if stage_cb: stage_cb("Cropping original blocks…")
        crop_original_data(item_list, image_3d_shape, lod_path, dtype, log, maximum_level,
                           progress_cb=progress_cb, stage_cb=stage_cb, abort_cb=abort_cb)

        # bottom up reconstruction
        if stage_cb: stage_cb("Downscaling / assembling…")
        assemble_level(lod_path, dtype, pad_image_shape, log, maximum_level - 1,
                       progress_cb=progress_cb, stage_cb=stage_cb, abort_cb=abort_cb)


        if stack_path is not None:
            tif.close()

        if progress_cb:
            progress_cb(100)
        if stage_cb:
            stage_cb("Finished all processing")



def main(file_path: str, dtype):
    if file_path.endswith(".csv"):
        # read CSV
        df = pd.read_csv(file_path, header=None)
        for i in df[0]:
            file_path_i = i
            print(f"[CSV Mode] Processing folder: {file_path_i}")
            oneVolProcessing(file_path_i, dtype)
    elif os.path.isdir(file_path):
        # read 1 folder of tiff
        print(f"[Folder Mode] Processing folder: {file_path}")
        oneVolProcessing(file_path, dtype)
    elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
        # read 1 tiff (3d stack)
        parent_dir = os.path.dirname(file_path)
        print(f"[3D Stack Mode] Processing single TIFF stack: {file_path}")
        oneVolProcessing(parent_dir, dtype, stack_path=file_path)
    else:
        raise ValueError("Unsupported input type. Provide .csv, folder, or single 3D TIFF path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D volume data into LOD octree structure.")
    parser.add_argument("file_path", help="Path to .csv, folder of TIFFs, or single 3D TIFF stack")
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "float32"],
                        help="Data type to use for processing (default: uint16)")
    parser.add_argument("--min-size", type=int, default=100, help="min size used for LOD estimation (default: 100)")
    args = parser.parse_args()

    dtype = getattr(np, args.dtype)
    main(args.file_path, dtype)