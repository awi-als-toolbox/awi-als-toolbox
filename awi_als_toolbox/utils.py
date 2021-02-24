# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

from pyproj import Geod


def geo_inverse(lon0, lat0, lon1, lat1):
    """ Inverse geodetic projection """
    g = Geod(ellps='WGS84')
    faz, baz, dist = g.inv(lon1, lat1, lon0, lat0)
    return faz, baz, dist
