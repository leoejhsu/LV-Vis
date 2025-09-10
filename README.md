# LV-Vis: Large-Volume LOD Visualization System

An open-source framework for **multi-resolution 3D volume visualization** built with **VisPy + PyQt5**. It enables efficient exploration of large volumetric datasets by combining preprocessing into octree-based Level-of-Detail (LOD) structures and interactive rendering with GPU acceleration.

---

## Features
- Preprocessing pipeline for TIFF stacks or folders into multi-level LOD format.
- Multi-volume viewer with interactive ROI selection, zoom-in/out, and alignment tools.
- GPU/CPU memory monitoring and camera controls built into the UI.
- GUI and CLI interfaces for both preprocessing and visualization.

---

## Installation

We recommend using Python ≥ 3.10 with a virtual environment.

You can choose either Conda or venv:

### Option 1: Conda 
```bash
conda create -n lvvis python=3.10 -y
conda activate lvvis

pip install -r requirements.txt
```

### Option 2: venv
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Dependencies include: `numpy`, `pandas`, `scikit-image`, `tifffile`, `matplotlib`, `tqdm`, `psutil`, `pynvml`, `imageio`, `PyQt5`, `vispy`, `pyopengl`.

---

## Usage

### Step 1: Preprocess Data
Convert raw TIFF datasets into LOD format.

**GUI Mode:**
```bash
python preprocess_gui.py
```

**CLI Mode:**
```bash
python preprocess.py --input /path/to/data --dtype float32 --min_size 100
```

See [docs/preprocessing.md](docs/preprocessing.md) for full details.

---

### Step 2: Visualize Data
Load preprocessed LOD data for interactive exploration.

**GUI Mode:**
```bash
python lv_vis_gui.py
```

**CLI Mode:**
```bash
python lv_vis.py --csv ./viewer.csv
```

See [docs/lv_vis.md](docs/lv_vis.md) for full details.

---

## Documentation
- [Preprocessing Guide](docs/preprocessing.md)
- [Visualization Guide](docs/lv_vis.md)

---
## Demo video
- Preprocessing: https://youtu.be/9aDNPccV26A
- Single Volume Visualization: https://youtu.be/lcOh9wv-mEM
- Multi-Volume Visualization: https://youtu.be/wzgscTEf6mM

---
## License
GPL-3.0-only. See [LICENSE](./LICENSE).  
Third-party attributions: see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

...
