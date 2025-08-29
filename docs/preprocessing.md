# Preprocessing Module Documentation

This module converts 3D imaging data (TIFF stacks or folders of slices) into an **octree-based Level-of-Detail (LOD)** structure for efficient multi-resolution visualization.

It provides two main entry points:
- **Command-line interface (CLI):** `preprocess.py`
- **Graphical User Interface (GUI):** `preprocess_gui.py`

---

## 1) Installation
Follow the [main README installation section](../README.md#installation) for environment setup instructions. 
This ensures consistent package versions and avoids duplication of installation steps.
---

## 2) Input Data Formats

Three input modes are supported:

1. **CSV Mode**  
   Each row specifies one volume folder or a single 3D TIFF file.
   ```csv
   /path/to/volume_folder_1
   /path/to/volume_folder_2
   /path/to/volume_stack.tiff
   ```
   ⚠️ Only the **first column (path)** is used during preprocessing. Any additional columns (e.g., coordinates for visualization) will be ignored.

2. **Folder Mode**  
   A folder containing a series of sequential 2D TIFF slices:
   ```
   volume_dir/
   ├── slice_0000.tif
   ├── slice_0001.tif
   └── ...
   ```

3. **Single 3D TIFF Mode**  
   A single multi-page TIFF stack file:
   ```
   kidney_volume.tiff
   ```

---

## 3) Output Structure

The program generates an **octree LOD folder** (default name `LOD_Data/`) under the input directory:

```
LOD_Data/
├── 0_0/
│   ├── 1_0/
│   │   ├── 2_0/
│   │   │   └── ...
│   │   └── 2_1/
│   └── 1_1/
│       └── ...
└── LOD_Data.npy   # merged root volume
```

---

## 4) Usage

### (A) CLI: `preprocess.py`

```bash
python preprocess.py --input INPUT_PATH --dtype DTYPE --lod_name LOD_Data --min_size 100
```

#### Arguments
- `--input INPUT_PATH`  
  Input source: CSV, folder of TIFFs, or single `.tif/.tiff`.
- `--dtype {uint16,float32}`  
  Output data type.
- `--lod_name LOD_Data`  
  Output LOD folder name (default: `LOD_Data`).
- `--min_size 100`  
  Minimum unit size for LOD estimation.

#### Examples
1. Batch from CSV:
   ```bash
   python preprocess.py --input ./multi_volumes.csv --dtype float32
   ```
2. Folder of TIFF slices:
   ```bash
   python preprocess.py --input ./hippo/ --dtype uint16
   ```
3. Single 3D TIFF stack:
   ```bash
   python preprocess.py --input ./kidney_stack.tiff --dtype float32 --lod_name MyLOD
   ```

---

### (B) GUI: `preprocess_gui.py`

Launch:
```bash
python preprocess_gui.py
```

The PyQt5 GUI workflow:

1. **Select Input Mode**  
   - CSV / Folder / Single 3D TIFF

2. **Set Parameters**  
   - **Data Type (dtype):** Auto / uint16 / float32  
   - **Output Folder Name (LOD):** default = `LOD_Data`  
   - **min_size:** for LOD estimation only (default = 100)

3. **Dry-Run (Estimate Levels)**  
   Quickly estimates maximum LOD levels and padded shape without writing files.

4. **Start Preprocessing**  
   Executes the full pipeline with live log updates.

5. **Cancel**  
   Requests cancellation (effective at the end of the current stage).

---

## 5) Logging

- **CLI:** outputs `LOD_log.txt` under the input folder.
- **GUI:** writes logs to `_preprocess_ui_logs/ui_log.txt` alongside the input data.

---

## 6) Notes

- Only the first column of CSV is used in preprocessing; coordinates are ignored.  
- The generated LOD hierarchy may consume significant memory and disk space.  
- Use **Dry-Run** to estimate levels and sizes before running the full job.  
- For visualization, prepare a **separate 4-column CSV** (`path,x,y,z`) to specify placement of volumes in world space.

---
