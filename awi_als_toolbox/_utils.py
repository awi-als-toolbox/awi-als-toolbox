# -*- coding: utf-8 -*-

"""
Small helper function for the toolbox
"""

__author__ = "Stefan Hendricks"

import yaml
import importlib
from pyproj import Geod
from attrdict import AttrDict


def geo_inverse(lon0, lat0, lon1, lat1):
    """ Inverse geodetic projection """
    g = Geod(ellps='WGS84')
    faz, baz, dist = g.inv(lon1, lat1, lon0, lat0)
    return faz, baz, dist


def get_yaml_cfg(yaml_filepath):
    """
    Return the content of a ymal config file as an AttrDict
    :param yaml_filepath: Path to yaml config file
    :return: AttrDict
    """
    with open(str(yaml_filepath), 'r') as fileobj:
        cfg = AttrDict(yaml.safe_load(fileobj))
    return cfg


def get_cls(module_name, class_name, relaxed=True):
    """ Small helper function to dynamically load classes"""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        if relaxed:
            return None
        else:
            raise ImportError("Cannot load module: %s" % module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        if relaxed:
            return None
        else:
            raise NotImplementedError("Cannot load class: %s.%s" % (module_name, class_name))