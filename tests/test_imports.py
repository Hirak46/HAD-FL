"""
HAD-FL | Basic Import & Smoke Tests
Run with: python3 -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_torch_import():
    import torch
    assert torch.__version__


def test_torchvision_import():
    import torchvision
    assert torchvision.__version__


def test_numpy_import():
    import numpy as np
    assert np.__version__


def test_pandas_import():
    import pandas as pd
    assert pd.__version__


def test_sklearn_import():
    import sklearn
    assert sklearn.__version__


def test_scipy_import():
    import scipy
    assert scipy.__version__


def test_hdbscan_import():
    import hdbscan
    assert hdbscan.__version__


def test_matplotlib_import():
    import matplotlib
    assert matplotlib.__version__


def test_seaborn_import():
    import seaborn
    assert seaborn.__version__


def test_src_files_exist():
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    assert os.path.exists(os.path.join(src_dir, 'fl_mnist.py'))
    assert os.path.exists(os.path.join(src_dir, 'fl_fmnist.py'))
    assert os.path.exists(os.path.join(src_dir, 'fl_cifar10.py'))


def test_results_dirs_creatable():
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()
    for d in ['mnist', 'fmnist', 'cifar10']:
        path = os.path.join(tmpdir, 'results', d)
        os.makedirs(path, exist_ok=True)
        assert os.path.isdir(path)
    shutil.rmtree(tmpdir)
