# -*- coding: utf-8 -*-

"""
The awi-als-toolobox python package is project to work with airborne laser scanner data
in a binary data format developed at the Alfred Wegener Institute Helmholtz Center for Polar
and Marine Research.
"""

__all__ = ["export", "filter", "graphics", "scripts",
           "AirborneLaserScannerFile", "AirborneLaserScannerFileV2", "ALSPointCloudData",
           "AlsDEM", "AlsDEMCfg", "ALSL4Grid", "ALSMergedGrid", "ALSGridCollection",
           "get_cls", "__version__"]

import sys
import importlib
from pathlib import Path

from ._bindata import (AirborneLaserScannerFile, AirborneLaserScannerFileV2, ALSPointCloudData)
from ._grid import (AlsDEM, AlsDEMCfg, ALSL4Grid, ALSMergedGrid, ALSGridCollection)
from ._utils import (get_yaml_cfg)

import warnings
warnings.filterwarnings("ignore")

# Get version from VERSION in package root
PACKAGE_ROOT_DIR = Path(__file__).absolute().parent
version_filepath = PACKAGE_ROOT_DIR / "VERSION"
try:
    with open(version_filepath) as f:
        version = f.read().strip()
except IOError:
    sys.exit("Cannot find VERSION file in package (expected: {}".format(version_filepath))


# Package Metadata
__version__ = version
__author__ = "Stefan Hendricks"
__author_email__ = "stefan.hendricks@awi.de"
