# -*- coding: utf-8 -*-

"""
"""

__all__ = ["export", "demgen", "graphics", "scripts",
           "AirborneLaserScannerFile", "AirborneLaserScannerFileV2"]

import os
import sys

from ._bindata import (AirborneLaserScannerFile, AirborneLaserScannerFileV2)

import warnings
warnings.filterwarnings("ignore")

# Get version from VERSION in package root
PACKAGE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
version_file = open(os.path.abspath(os.path.join(PACKAGE_ROOT_DIR, "VERSION")))
try:
    with version_file as f:
        version = f.read().strip()
except IOError:
    sys.exit("Cannot find VERSION file in package (expected: {}".format(version_file))


# Package Metadata
__version__ = version
__author__ = "Stefan Hendricks"
__author_email__ = "stefan.hendricks@awi.de"
