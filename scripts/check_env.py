#!/usr/bin/env python3
"""
HAD-FL | Environment Checker
Run this before experiments to verify all dependencies are installed.
Usage: python3 scripts/check_env.py
"""

import sys
import importlib

REQUIRED = [
    ("torch",        "PyTorch"),
    ("torchvision",  "TorchVision"),
    ("numpy",        "NumPy"),
    ("pandas",       "Pandas"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "SciPy"),
    ("hdbscan",      "HDBSCAN"),
    ("matplotlib",   "Matplotlib"),
    ("seaborn",      "Seaborn"),
]

print("=" * 55)
print("  HAD-FL  |  Environment Check")
print("=" * 55)

all_ok = True
for module, name in REQUIRED:
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "unknown")
        print(f"  [OK]  {name:<18} {version}")
    except ImportError:
        print(f"  [MISSING]  {name}")
        all_ok = False

print("-" * 55)

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"        Memory: {mem:.1f} GB")
    else:
        print("  [CPU] No CUDA GPU detected — will run on CPU (slower)")
except Exception:
    pass

print("=" * 55)

if all_ok:
    print("  All dependencies satisfied. Ready to run!")
    sys.exit(0)
else:
    print("  Some packages are missing. Run:  pip install -r requirements.txt")
    sys.exit(1)
