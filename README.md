# Viewer + DataProcessing for Various DeepLearning Models

Core Features:
- Automate depth estimation over a large dataset
- View and assess the model being interfaced

Viewer Features:
- Unified Drag & Zoom across all renders 

# Installation

This was tested on Linux with an RTX3090 and Cuda 12.4.

```
conda env create -f environment.yml
conda activate DLviewer

# For UniDepth V2
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth/
pip install -e .
python ./scripts/demo.py

## Possible solution to issue with libstdc++.so
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

```