#!/bin/bash
# Activate the Mu2e Python environment for the GNN clustering project.
# Source this file, do not execute it: source setup_env.sh

source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
# pyenv is a shell function defined by setupmu2e-art.sh; in non-interactive
# shells it may not be exported, so fall back to sourcing activate directly.
if declare -f pyenv &>/dev/null; then
    pyenv ana 2.6.1
else
    source /cvmfs/mu2e.opensciencegrid.org/env/ana/2.6.1/bin/activate
fi

# User-installed packages (torch_geometric lives here)
export PYTHONPATH="/nashome/w/wzhou2/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# Add project src to Python path so imports like "from data.dataset import ..."  work
export PYTHONPATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/src:${PYTHONPATH}"

echo "Environment ready: Python $(python3 --version 2>&1 | cut -d' ' -f2), torch $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
