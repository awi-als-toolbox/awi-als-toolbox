# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

from pyproj import Proj

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.filters import maximum_filter


class AlsDEM(object):
    """ TODO: Documentation """

    MAXIMUM_FILTER_DEFAULT = {'size': 3, 'mode': 'nearest'}
    GRIDDATA_DEFAULT = {'method': 'linear', 'rescale': False}

    def __init__(self, **kwargs):
        """ TODO: Documentation """
        self._has_data = False
        self._align_heading = False
        self._use_maximum_filter = True
        self._resolution = 1.0
        self._grid_pad_fraction = 0.01
        self._dem_settings = self.GRIDDATA_DEFAULT
        self._dem_settings.update(kwargs)
        self._maximum_filter_kwargs = self.MAXIMUM_FILTER_DEFAULT

    def set_data(self, data):
        """ TODO: Documentation """
        self.longitude = data.longitude
        self.latitude = data.latitude
        self.elevation = data.elevation
        self._has_data = True

    def set_resolution(self, resolution):
        """ TODO: Documentation """
        self._resolution = resolution

    def set_maximum_filter(self, onoff, **kwargs):
        """
        TODO: this should have an automatic method to decide
        automatically what filter width to use, based on
        scanner point spacing and DEM resolution
        """
        self._use_maximum_filter = onoff
        self._maximum_filter_kwargs.update(kwargs)

    def set_align_heading(self, onoff):
        """ TODO: Documentation """
        self._align_heading = onoff

    def griddata(self, method=None, rescale=None):
        """ Grids irregular laser scanner points to regular grid """
        # TODO: Properly validate data
        self._proj()
        if self._align_heading:
            self._align()
        self._griddata()
        if self._use_maximum_filter:
            self._maximum_filter()

    def _proj(self):
        """ Calculate projection coordinates """

        # Guess projection center
        lat_0ts = np.nanmedian(self.latitude)
        lon_0 = np.nanmedian(self.longitude)

        # Get the nan mask (joint mask of longitude & latitude)
        is_nan = np.logical_or(np.isnan(self.longitude), np.isnan(self.latitude))
        nan_mask = np.where(is_nan)

        lon, lat = np.copy(self.longitude), np.copy(self.latitude)
        lon[nan_mask] = lon_0
        lat[nan_mask] = lat_0ts

        # get projection coordinates
        p = Proj(proj='stere', lat_ts=lat_0ts, lat_0=lat_0ts, lon_0=lon_0)
        self.x, self.y = p(lon, lat)

        if len(nan_mask) > 0:
            self.x[nan_mask] = np.nan
            self.y[nan_mask] = np.nan

        # import matplotlib.pyplot as plt
        #
        # plt.figure("lon", dpi=300)
        # plt.imshow(lon)
        #
        # plt.figure("lat", dpi=300)
        # plt.imshow(lat)
        #
        # plt.figure("x", dpi=300)
        # plt.imshow(self.x)
        #
        # plt.figure("y", dpi=300)
        # plt.imshow(self.y)
        #
        # plt.show()
        # stop


    def _griddata(self):
        """ Do the actual gridding """
        res = self._resolution
        # Get area of data
        xmin = np.nanmin(self.x)
        xmax = np.nanmax(self.x)
        ymin = np.nanmin(self.y)
        ymax = np.nanmax(self.y)
        # Add padding
        width = xmax-xmin
        height = ymax-ymin
        pad = np.amax([self._grid_pad_fraction*width, self._grid_pad_fraction*height])
        xmin = np.floor(xmin - pad)
        xmax = np.ceil(xmax + pad)
        ymin = np.floor(ymin - pad)
        ymax = np.ceil(ymax + pad)

        # Create Grid and no data mask
        self.lrx = np.arange(xmin, xmax+res, res)
        self.lry = np.arange(ymin, ymax+res, res)
        self.dem_x, self.dem_y = np.meshgrid(self.lrx, self.lry)
        self.nonan = np.where(np.logical_or(np.isfinite(self.x), np.isfinite(self.y)))

        # Create regular grid
        self.dem_z = griddata((self.x[self.nonan].flatten(), self.y[self.nonan].flatten()),
                              self.elevation[self.nonan].flatten(),
                              (self.dem_x, self.dem_y),
                              **self._dem_settings)
        self.dem_z = np.ma.array(self.dem_z)
        self.dem_mask = np.zeros(self.dem_z.shape, dtype=np.bool)

    def _maximum_filter(self):
        """
        Remove interpolation results in areas where no als data is available
        but which are in the concex hull of the swath
        """
        res = self._resolution
        xedges = np.linspace(self.lrx[0]-res/2.,
                             self.lrx[-1]+res/2.0, len(self.lrx)+1)
        yedges = np.linspace(self.lry[0]-res/2.,
                             self.lry[-1]+res/2.0, len(self.lry)+1)

        # Calculates point density of als shots per DEM grid cell
        self.rzhist, xe, ye = np.histogram2d(self.x[self.nonan].flatten(),
                                             self.y[self.nonan].flatten(),
                                             bins=[xedges, yedges])
        self.rzhist = self.rzhist.transpose()
        data_mask = self.rzhist > 0.0

        data_mask = maximum_filter(data_mask, **self._maximum_filter_kwargs)

#        structure = [[0,1,0],
#                     [1,1,1],
#                     [0,1,0]]
#        cluster_id, num = measurements.label(nodata_mask, structure=structure)
#        cluster_size = measurements.sum(nodata_mask, cluster_id,
#                                        index=np.arange(cluster_id.max()+1))
#        data_mask = cluster_size[cluster_id] < 50
        self.dem_mask = ~data_mask

    def _align(self):
        """
        Rotates DEM that mean flight direction
        """

        shape = np.shape(self.x)

        # Get angle of direction (cbi: center beam index)
        # NOTE: This implementation seems to be unstable, because the shot with the center beam index can be NaN
        # cbi = np.median(np.arange(len(self.x[0, :]))).astype(int)
        # vec1 = [self.x[0, cbi], self.y[0, cbi],  0.0]
        # vec2 = [self.x[-1, cbi], self.y[-1, cbi], 0.0]

        # Alternative implementation with mean over all entries within the line.
        # -> should be a good approximation of the line center
        vec1 = [np.nanmedian(self.x[0, :]), np.nanmedian(self.y[0, :]), 0.0]
        vec2 = [np.nanmedian(self.x[-1, :]), np.nanmedian(self.y[-1, :]), 0.0]

        angle = -1.0*np.arctan((vec2[1]-vec1[1])/(vec2[0]-vec1[0]))

        # Get center point
        xc = np.nanmedian(self.x)
        yc = np.nanmedian(self.y)

        # Reform points
        points = [self.x.flatten()-xc, self.y.flatten()-yc]

        # Execute the rotation
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        points_rotated = rot_matrix.dot(points)
        self.x = np.reshape(points_rotated[0, :], shape)
        self.y = np.reshape(points_rotated[1, :], shape)

        # Save conversion parameters for reuse
        self._align_parameters = {'center_point': (xc, yc),
                                  'angle': angle,
                                  'rotation_matrix': rot_matrix}