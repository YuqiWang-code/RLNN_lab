#!/usr/bin/env bash
set -euo pipefail
# Run inside the dedicated Python 3.10 RLNN virtual environment.
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install torch_scatter --only-binary=:all: -f https://data.pyg.org/whl/torch-2.9.0+cu130.html
python -m pip install torch_geometric==2.7.0 accelerate==1.0.1 networkx==2.8.8 numpy==1.26.4 scipy==1.13.1 tqdm==4.67.1 psutil
python -m pip freeze
