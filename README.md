# PySpatial: A High-Speed Whole Slide Image Pathomics Toolkit

**A streamlined toolkit for direct WSI-level feature extraction, leveraging spatial indexing and matrix-based computation for up to 10× speedups on small objects and 2× on larger ones.**

Paper: [arXiv:2501.06151](https://arxiv.org/abs/2501.06151)

---

## Overview

Whole Slide Image (WSI) analysis plays a crucial role in modern digital pathology, enabling large-scale feature extraction from tissue samples. Traditional pipelines (e.g., CellProfiler) require patch-level segmentation, per-patch feature extraction, and coordinate remapping—introducing significant overhead. PySpatial eliminates these intermediate steps by operating directly on annotated computational regions of interest. It combines R-tree spatial indexing with high-throughput matrix-based batch computation to accelerate feature extraction while preserving spatial context and accuracy.

---

## Dependencies & Prerequisites

- **Ubuntu / Debian**  
- **Conda** (to create the provided `environment.yml`)  
- System packages:
  ```bash
  sudo apt install default-jre
  sudo apt-get install libmysqlclient-dev
  sudo apt-get install dpkg-dev build-essential python3-dev \
      freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev \
      libgstreamer-plugins-base1.0-dev libgtk-3-dev \
      libjpeg-dev libnotify-dev libpng-dev libsdl2-dev \
      libsm-dev libtiff-dev

---
## Usage
### PEC dataset feature extraction
  ```bash
  python PEC_extract_feature_main.py
```

### KPMP dataset feature extraction
```bash
  python KPMP_extract_feature_main.py
```
---
## Data
Please replace the "example" folder with this [link](https://vanderbilt.box.com/s/29ceofv5fd0qsxl10iprpsl1zosdi7q1)    

