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
        self.metadata = als.metadata.copy()
        self.processing_level = "Level-3 Collated (l3c)"
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

        # Project lon/lat coordinates of shots into cartesian coordinate system
        self._proj()

        # if self.cfg.align_heading:
        #     self._align()

        # Grid all shots
        self._griddata()

        # Create grid statistics
        self._grid_statistics()

        # Interpolate gaps
        # NOTE: This procedure raises the processing level from 3 to 4
        if self.cfg.gap_filter["algorithm"] != "none":
            self._gap_filter()

        # Manage metadata
        self._update_metadata()

    def get_swath_lonlat_center(self):
        """
        Get the center position (longitude, latitude) of the swath segment
        :return: (float, float) lon_0, lat_0
        """
        # Guess projection center
        lat_0 = np.nanmedian(self.als.latitude)
        lon_0 = np.nanmedian(self.als.longitude)
        return lon_0, lat_0

    def _proj(self):
        """ Calculate projection coordinates """

        # TODO: Add option to prescribe projection

        # Guess projection center
        lon_0, lat_0ts = self.get_swath_lonlat_center()

        # Get the nan mask (joint mask of longitude & latitude)
        is_nan = np.logical_or(np.isnan(self.als.longitude), np.isnan(self.als.latitude))
        nan_mask = np.where(is_nan)

        lon, lat = np.copy(self.als.longitude), np.copy(self.als.latitude)

        lon[nan_mask] = lon_0
        lat[nan_mask] = lat_0ts

        # get projection coordinates
        if self.cfg.projection == "auto":
            self._proj_parameters = dict(proj='stere', lat_ts=lat_0ts, lat_0=lat_0ts, lon_0=lon_0, ellps="WGS84")
        else:
            self._proj_parameters = self.cfg.projection
        self.p = Proj(**self._proj_parameters)
        self.x, self.y = self.p(lon, lat)

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
        self.xc = np.arange(xmin, xmax+res, res)
        self.yc = np.arange(ymin, ymax+res, res)
        self.dem_x, self.dem_y = np.meshgrid(self.xc, self.yc)
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

        # Compute lons, lats for grid
        self.lon, self.lat = self.p(self.dem_x, self.dem_y, inverse=True)

    def _grid_statistics(self):
        """ Compute grid statistics (Currently number of shots per grid cell only"""

        res = self.cfg.resolution
        xedges = np.linspace(self.xc[0]-res/2., self.xc[-1]+res/2.0, len(self.xc)+1)
        yedges = np.linspace(self.yc[0]-res/2., self.yc[-1]+res/2.0, len(self.yc)+1)

        # Calculates point density of als shots per DEM grid cell
        rzhist, xe, ye = np.histogram2d(self.x[self.nonan].flatten(),
                                        self.y[self.nonan].flatten(),
                                        bins=[xedges, yedges])
        self.n_shots = rzhist.transpose()

    def _gap_filter(self):
        """
        Remove interpolation results in areas where no als data is available
        but which are in the concex hull of the swath
        """

        self.processing_level = "Level-4 (l4)"

        data_mask = self.n_shots > 0.0

        filter_algorithm = self.cfg.gap_filter["algorithm"]
        if filter_algorithm == "maximum_filter":
            data_mask = maximum_filter(data_mask, **self.cfg.gap_filter["keyw"])
        else:
            raise NotImplementedError("Filter algorithm: %s" % filter_algorithm)

        self.dem_mask = ~data_mask

    def _align(self):
        """
        Rotates DEM that mean flight direction
        """

        shape = np.shape(self.x)

        # Get the rotation angle
        # NOTE: Heading describes the angle w.r.t. to the positive y-axis in projection coordinates
        #       We want to rotate the points that aircraft heading is rotated to heading 90 -> positive x-axis
        angle = self.heading_prj - 0.5*np.pi

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

    def _update_metadata(self):
        """ Update the metadata object with specifics for gridded products """

        # Data is now on a space-time grid
        self.metadata.set_attribute("cdm_data_type", "grid")
        self.metadata.set_attribute("processing_level", self.processing_level)
        self.metadata.set_attribute("geospatial_bounds_crs", "EPSG:54026")
        self.metadata.set_attribute("geospatial_lon_units", "m")
        self.metadata.set_attribute("geospatial_lat_units", "m")
        self.metadata.set_attribute("geospatial_lon_resolution", self.cfg.resolution)
        self.metadata.set_attribute("geospatial_lat_resolution", self.cfg.resolution)

    @property
    def heading_prj(self):
        """ The heading of the track in the current projection """

        # Get angle of direction (cbi: center beam index)
        # NOTE: This implementation seems to be unstable, because the shot with the center beam index can be NaN
        # cbi = np.median(np.arange(len(self.x[0, :]))).astype(int)
        # vec1 = [self.x[0, cbi], self.y[0, cbi],  0.0]
        # vec2 = [self.x[-1, cbi], self.y[-1, cbi], 0.0]

        # Alternative implementation with mean over all entries within the line.
        # -> should be a good approximation of the line center
        # NOTE: 2019-05-30: Relaxed the criterion even further (mean of first and last 10 scan lines)
        # vec1 = [np.nanmedian(self.x[0:10, :]), np.nanmedian(self.y[0:10, :]), 0.0]
        # vec2 = [np.nanmedian(self.x[-11:-1, :]), np.nanmedian(self.y[-11:-1, :]), 0.0]
        # return np.arctan((vec2[1]-vec1[1])/(vec2[0]-vec1[0]))

        # Third implementation (calculate a header for each shot per line and use average)
        n_lines, n_shots_per_line = np.shape(self.x)
        angles = np.full((n_shots_per_line), np.nan)
        for shot_index in np.arange(n_shots_per_line):
            p0 = [self.x[0, shot_index], self.y[0, shot_index]]
            p1 = [self.x[-1, shot_index], self.y[-1, shot_index]]
            angles[shot_index] = np.arctan2((p1[1]-p0[1]), (p1[0]-p0[0]))

        # Angles are with respect to positive x-axis
        # Assumption positive y is north -> reference to positive y
        return 0.5*np.pi-np.nanmean(angles)

    @property
    def max_side_len(self):
        """
        Computes the maximum from [width, height] of the gridded DEM
        :return:
        """

        height = np.nanmax(self.dem_y) - np.nanmax(self.dem_y)
        width = np.nanmax(self.dem_x) - np.nanmax(self.dem_x)

        return np.nanmax([height, width])

    @property
    def resolution(self):
        """
        The resolution of the DEM in meters
        :return:
        """
        return self.cfg.resolution

    @property
    def fn_proc_level(self):
        """
        A filename compatible processing level str
        :return: str
        """
        for proc_level_id in ["l3c", "l4"]:
            if proc_level_id in self.processing_level:
                return proc_level_id
        return ""

    @property
    def fn_res(self):
        """
        A filename compatible resolution str
        :return: str
        """
        res_str = "%.2fm" % self.resolution
        res_str = res_str.replace(".", "p")
        return res_str

    @property
    def fn_tcs(self):
        """
        A filename compatible time coverage start str
        :return: str
        """
        datetime_format = ("%Y%m%dT%H%M%S")
        return self.als.tcs_segment_datetime.strftime(datetime_format)

    @property
    def fn_tce(self):
        """
        A filename compatible time coverage end str
        :return: str
        """
        datetime_format = ("%Y%m%dT%H%M%S")
        return self.als.tce_segment_datetime.strftime(datetime_format)

    @property
    def ref_time(self):
        return self.als.ref_time

    @property
    def time_bnds(self):
        return self.als.time_bnds


class AlsDEMCfg(object):

    def __init__(self, resolution=1.0, method="default", gap_filter="default", grid_pad_fraction=0.01,
                 segment_len_secs=20.0, projection="auto"):
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
        self._resolution = resolution

        # Properties for data gridding
        if method == "default":
            method = dict(algorithm="scipy.griddata", keyw=dict(method="linear", rescale=True))
        self._griddata = method

        # Method do properly handle data gaps after gridding
        if gap_filter == "default":
            gap_filter = dict(algorithm="maximum_filter", keyw=dict(size=3, mode="nearest"))
        self._gap_filter = gap_filter

        # Padding of the grid extent
        self._grid_pad_fraction = grid_pad_fraction

        # Standard length of segments
        self._segment_len_secs = segment_len_secs

        # Projection information
        # default is "auto", meaning that the projection will be estimated from the point cloud.
        # Else, needs to be a dictionary with valid input for pyproj.Proj
        self.projection = projection

    @classmethod
    def preset(cls, mode, **kwargs):
        """
        Return defined presets for data gridding
        :param mode: (str) Name of the mode (currently only `sea_ice_low`)
        :return:
        """

        valid_presets = ["sea_ice_low, sea_ice_high"]

        # Low altitude (200 - 1000 ft) sea ice surveys
        # -> high latitude resolution
        if str(mode) == "sea_ice_low":
            keyw = dict(resolution=0.25, segment_len_secs=60)

        # High altitude (> 1500 ft) sea ice surveys
        # -> default settings
        elif str(mode) == "sea_ice_high":
            keyw = dict(resolution=0.5, segment_len_secs=60)

        else:
            msg = "Unknown preset: %s (known presets: %s)" % (str(mode), ",".join(valid_presets))
            raise ValueError(msg)

        keyw.update(kwargs)
        cfg = cls(**keyw)

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

    @property
    def segment_len_secs(self):
        return int(self._segment_len_secs)

