#!/bin/bash
# Activate the Mu2e Python environment for the GNN clustering project.
# Source this file, do not execute it: source setup_env.sh

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana 2.6.1

# Add project src to Python path so imports like "from data.dataset import ..."  work
export PYTHONPATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/src:${PYTHONPATH}"

echo "Environment ready: Python $(python3 --version 2>&1 | cut -d' ' -f2), torch $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
