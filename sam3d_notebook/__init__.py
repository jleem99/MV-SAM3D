"""sam3d_notebook – installable re-export of MV-SAM3D ``notebook/`` helpers.

This package makes the MV-SAM3D demo/notebook utilities importable without
``sys.path`` hacks.  The env-var guards below run *before* any submodule
(e.g. ``sam3d_notebook.inference``) is loaded, preventing the ``KeyError``
that the upstream ``inference.py`` would raise outside Conda.
"""

from __future__ import annotations

import os

# upstream ``notebook/inference.py`` unconditionally does
#   os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
# which crashes when there is no Conda environment active.
# Pre-set both before any submodule import.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
