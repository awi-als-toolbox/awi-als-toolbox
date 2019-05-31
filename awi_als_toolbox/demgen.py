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

    def __init__(self, als, cfg=None):
        """
        Create a gridded DEM from point cloud airborne laser scanner (ALS) data
        :param als: awi_als_toolbox.reader.ALSData object
        :param cfg:
        """

        self.als = als
        if cfg is None:
            cfg = AlsDEMCfg()
        self.cfg = cfg

    @property
    def dem_z_masked (self):
        dem_z_masked = np.copy(self.dem_z)
        dem_z_masked[self.dem_mask] = np.nan
        return dem_z_masked


    def create(self):
        """ Grids irregular laser scanner points to regular grid """
        # TODO: Properly validate data
        self._proj()
        if self.cfg.align_heading:
            self._align()
        self._griddata()
        if self.cfg.gap_filter["algorithm"] != "none":
            self._gap_filter()

    def _proj(self):
        """ Calculate projection coordinates """

        # Guess projection center
        lat_0ts = np.nanmedian(self.als.latitude)
        lon_0 = np.nanmedian(self.als.longitude)

        # Get the nan mask (joint mask of longitude & latitude)
        is_nan = np.logical_or(np.isnan(self.als.longitude), np.isnan(self.als.latitude))
        nan_mask = np.where(is_nan)

        lon, lat = np.copy(self.als.longitude), np.copy(self.als.latitude)
        lon[nan_mask] = lon_0
        lat[nan_mask] = lat_0ts

        # get projection coordinates
        p = Proj(proj='stere', lat_ts=lat_0ts, lat_0=lat_0ts, lon_0=lon_0)
        self.x, self.y = p(lon, lat)

        if len(nan_mask) > 0:
            self.x[nan_mask] = np.nan
            self.y[nan_mask] = np.nan

    def _griddata(self):
        """ Do the actual gridding """
        res = self.cfg.resolution

        # Get area of data
        xmin, xmax = np.nanmin(self.x), np.nanmax(self.x)
        ymin, ymax = np.nanmin(self.y),  np.nanmax(self.y)

        # Add padding
        width = xmax-xmin
        height = ymax-ymin
        pad = np.amax([self.cfg.grid_pad_fraction*width, self.cfg.grid_pad_fraction*height])
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
        gridding_algorithm = self.cfg.griddata["algorithm"]
        if gridding_algorithm == "scipy.griddata":
            self.dem_z = griddata((self.x[self.nonan].flatten(), self.y[self.nonan].flatten()),
                                  self.als.elevation[self.nonan].flatten(),
                                  (self.dem_x, self.dem_y),
                                  **self.cfg.griddata["keyw"])
        else:
            raise NotImplementedError("Gridding algorithm: %s" % gridding_algorithm)

        self.dem_z = np.ma.array(self.dem_z)
        self.dem_mask = np.zeros(self.dem_z.shape, dtype=np.bool)

    def _gap_filter(self):
        """
        Remove interpolation results in areas where no als data is available
        but which are in the concex hull of the swath
        """
        res = self.cfg.resolution
        xedges = np.linspace(self.lrx[0]-res/2., self.lrx[-1]+res/2.0, len(self.lrx)+1)
        yedges = np.linspace(self.lry[0]-res/2., self.lry[-1]+res/2.0, len(self.lry)+1)

        # Calculates point density of als shots per DEM grid cell
        self.rzhist, xe, ye = np.histogram2d(self.x[self.nonan].flatten(),
                                             self.y[self.nonan].flatten(),
                                             bins=[xedges, yedges])
        self.rzhist = self.rzhist.transpose()
        data_mask = self.rzhist > 0.0

        filter_algorithm = self.cfg.gap_filter["algorithm"]
        if filter_algorithm == "maximum_filter":
            data_mask = maximum_filter(data_mask, **self.cfg.gap_filter["keyw"])
        else:
            raise NotImplementedError("Filter algorithm: %s" % filter_algorithm)

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
        # NOTE: 2019-05-30: Relaxed the criterion even further (mean of first and last 10 scan lines)
        vec1 = [np.nanmedian(self.x[0:10, :]), np.nanmedian(self.y[0:10, :]), 0.0]
        vec2 = [np.nanmedian(self.x[-11:-1, :]), np.nanmedian(self.y[-11:-1, :]), 0.0]
        angle = -1.0*np.arctan((vec2[1]-vec1[1])/(vec2[0]-vec1[0]))

        # validity check -> Do not rotate if angle is nan
        if np.isnan(angle):
            return

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


class AlsDEMCfg(object):

    def __init__(self, resolution=None, align_heading=None, griddata=None, gap_filter=None, grid_pad_fraction=None):
        """
        Filter settings for DEM generation
        :param resolution:
        :param align_heading:
        :param griddata:
        :param gap_filter:
        :param grid_pad_fraction:
        """

        # --- Set Default settings ---

        # DEM resolution in meter
        if resolution is None:
            resolution = 1.0
        self._resolution = resolution

        # Align heading (on be default)
        # TODO: Allow to specify target direction
        if align_heading is None:
            align_heading = True
        self._align_heading = align_heading

        # Properties for data gridding
        if griddata is None:
            griddata = dict(algorithm="scipy.griddata", keyw=dict(method="linear", rescale=True))
        self._griddata = griddata

        # Method do properly handle data gaps after gridding
        if gap_filter is None:
            gap_filter = dict(algorithm="maximum_filter", keyw=dict(size=3, mode="nearest"))
        self._gap_filter = gap_filter

        # Padding of the grid extent
        if grid_pad_fraction is None:
            grid_pad_fraction = 0.01
        self._grid_pad_fraction = grid_pad_fraction

    @classmethod
    def preset(cls, mode):
        """
        Return defined presets for data gridding
        :param mode: (str) Name of the mode (currently only `sea_ice_low`)
        :return:
        """

        valid_presets = ["sea_ice_low"]

        if str(mode) == "sea_ice_low":
            cfg = cls(resolution=0.25)
        else:
            msg = "Unknown preset: %s (known presets: %s)" % (str(mode), ",".join(valid_presets))
            raise ValueError(msg)

        return cfg

    @property
    def resolution(self):
        return float(self._resolution)

    @property
    def align_heading(self):
        return bool(self._align_heading)

    @property
    def griddata(self):
        return dict(self._griddata)

    @property
    def gap_filter(self):
        return dict(self._gap_filter)

    @property
    def grid_pad_fraction(self):
        return float(self._grid_pad_fraction)

