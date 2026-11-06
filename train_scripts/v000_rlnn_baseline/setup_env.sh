#!/usr/bin/env bash
set -euo pipefail
# Run inside a new Python 3.11 virtual environment on the Linux server.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch_scatter --only-binary=:all: -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
python -m pip install torch_geometric==2.6.1 accelerate==1.0.1 networkx==2.8.8 numpy==1.26.4 scipy==1.13.1 tqdm==4.67.1
python -m pip freeze
