# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import os
import xarray as xr
from netCDF4 import num2date, date2num

import numpy as np

import pyproj
from osgeo import gdal, osr

from loguru import logger

from scipy.interpolate import griddata
from scipy.ndimage.filters import maximum_filter

from ._utils import get_yaml_cfg, geo_inverse


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
        self.p = pyproj.Proj(**self._proj_parameters)
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
        angles = np.full(n_shots_per_line, np.nan)
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
        datetime_format = "%Y%m%dT%H%M%S"
        return self.als.tcs_segment_datetime.strftime(datetime_format)

    @property
    def fn_tce(self):
        """
        A filename compatible time coverage end str
        :return: str
        """
        datetime_format = "%Y%m%dT%H%M%S"
        return self.als.tce_segment_datetime.strftime(datetime_format)

    @property
    def ref_time(self):
        return self.als.ref_time

    @property
    def time_bnds(self):
        return self.als.time_bnds

    @property
    def grid_mapping_items(self):
        name, attrs = None, {}
        if self.cfg.grid_mapping is not None:
            name, attrs = self.cfg.grid_mapping["name"], self.cfg.grid_mapping["attrs"]
        return name, attrs


class AlsDEMCfg(object):

    def __init__(self, input_filter=None, connect_keyw=None, resolution_m=1.0, segment_len_secs=30, method=None, gap_filling=None,
                 grid_pad_fraction=0.05, projection=None, grid_mapping=None):
        """
        Settings for DEM generation including spatial and temporal resolution, gridding settings and
        target projection
        :param input_filter:
        :param resolution_m:
        :param segment_len_secs:
        :param method:
        :param gap_filling:
        :param grid_pad_fraction:
        :param projection:
        :param grid_mapping:
        """

        # --- Set Default settings ---

        # DEM resolution in meter
        self.resolution = resolution_m

        # Lengths of the segments
        self.segment_len_secs = segment_len_secs

        # A list of filters applied to the input data
        # before gridding.
        self.input_filter = input_filter

        if connect_keyw is None:
            connect_keyw = {}
        self.connect_keyw = connect_keyw

        # Properties for data gridding
        if method is None:
            method = dict(algorithm="scipy.griddata", keyw=dict(method="linear", rescale=True))
        self.griddata = method

        # Method to handle data gaps
        self.gap_filter = gap_filling

        # Padding of the grid extent
        self.grid_pad_fraction = grid_pad_fraction

        # Projection information
        # default is "auto", meaning that the projection will be estimated from the point cloud.
        # Else, needs to be a dictionary with valid input for pyproj.Proj
        self.projection = projection

        # Grid Mapping
        # same information as projection, but in a format for netCDF grid mapping variable
        self.grid_mapping = grid_mapping

    @classmethod
    def from_cfg(cls, yaml_filepath):
        """
        Initialize the ALSDEMCfg instance from a yaml config file w
        :param yaml_filepath:
        :return:
        """

        # Read the yaml file
        cfg = get_yaml_cfg(yaml_filepath)
        return cls(**cfg)

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

        # High altitude (> 1500 ft) sea ice surveys
        # -> default settings
        elif str(mode) == "mosaic_standard":
            keyw = dict(resolution=0.5, segment_len_secs=30)

        else:
            msg = "Unknown preset: %s (known presets: %s)" % (str(mode), ",".join(valid_presets))
            raise ValueError(msg)

        keyw.update(kwargs)
        cfg = cls(**keyw)

        return cfg


class ALSGridCollection(object):

    def __init__(self, filepaths, res=None, ignore_list=None):
        self.filepaths = filepaths
        self.res = res
        self.ignore_list = ignore_list
        self.grids = []
        self._read_grid_data()

    def add_drift_correction_reference(self, ref):
        """
        Add reference data to correct for ice drift
        :param ref: Any object with time (UTC datetime), longitude and latitude attributes
        :return:
        """

        logger.info("Apply drift correction")
        self.ref = ref

        # Coverage test
        logger.info("Check temporal coverage of drift correction data")
        time_bounds = self.time_bounds
        in_range = np.logical_and(ref.time >= time_bounds[0], ref.time <= time_bounds[1])
        if len(np.where(in_range)[0]) < self.n_grids:
            raise ValueError("Not all grids in drift correction time range")
        logger.info("-> Test passed")

        # Compute the displacement for each grid segment in range resolution (pixel) units
        # NOTE: This needs to be done in the projection of the grids

        # Step 1: Get lon/lat positions of reference station for grid reference times
        ref_lons, ref_lats, times_num, time_num_ref = self._get_ref_lonlat(ref)

        # Step 2: Convert reference station lon/lats to same grid
        p = pyproj.Proj(self.proj4str)
        ref_x, ref_y = p(ref_lons, ref_lats)

        # Step 3: Compute displacement in spatial resolution units
        ref_x_delta, ref_y_delta = ref_x - ref_x[0], ref_y - ref_y[0]
        grid_offset_x, grid_offset_y = ref_x_delta, ref_y_delta
        grid_offset_x = grid_offset_x.astype(int)
        grid_offset_y = grid_offset_y.astype(int)

        # Step 4: Apply offset to grid
        for i, (grid, offset_x, offset_y) in enumerate(zip(self.grids, grid_offset_x, grid_offset_y)):
            msg = "Time Offset: %.0f sec, displacement: (%.0fm, %.0fm)"
            timedelta_secs = times_num[i] - times_num[0]
            msg = msg % (timedelta_secs, offset_x, offset_y)
            logger.debug(msg)
            grid.offset = [-offset_x, -offset_y]

    def set_maximum_dist2ref(self, max_dist):
        """
        Select number of grids based on their distance to the reference target
        :param max_dist:
        :return:
        """
        logger.info("Apply max dist filter to segments")

        # Step 1: Get lon/lat positions of reference station for grid reference times
        ref_lons, ref_lats, _, _ = self._get_ref_lonlat(self.ref)

        for i, grid in enumerate(self.grids):
            grid_lon, grid_lat = np.nanmean(grid.lons), np.nanmean(grid.lats)
            _, _, dist = geo_inverse(grid_lon, grid_lat, ref_lons[i], ref_lats[i])
            logger.debug("Grid %d: dist to ref: %.2fm" % (i+1, dist))
            if dist > max_dist:
                self.ignore_list.append(i)

    def _read_grid_data(self):
        for filepath in self.filepaths:
            logger.info("Read: %s" % os.path.split(filepath)[-1])
            grid_data = ALSL4Grid(filepath)
            self.grids.append(grid_data)

    def get_merged_grid(self):
        x_min, x_max = self.xc_bounds
        y_min, y_max = self.yc_bounds
        merged_grid = ALSMergedGrid(x_min, x_max, y_min, y_max, self.res, self.proj4str)
        logger.info("Merge Grids:")
        for i, grid in enumerate(self.grids):
            if i in self.ignore_list:
                continue
            logger.info("... %g / %g done [ref_time:%s]" % (i+1, self.n_grids, grid.reftime))
            merged_grid.add_grid(grid)
        logger.info("... %g / %g done" % (self.n_grids, self.n_grids))
        return merged_grid

    def _get_ref_lonlat(self, ref):
        times_num = date2num(self.times, "seconds since 1970-01-01")
        time_num_ref = date2num(ref.time, "seconds since 1970-01-01")
        ref_lons = np.interp(times_num, time_num_ref, ref.longitude)
        ref_lats = np.interp(times_num, time_num_ref, ref.latitude)
        return ref_lons, ref_lats, times_num, time_num_ref

    @property
    def xc_bounds(self):
        grid_min_bounds = [grid.xc_bounds[0] for grid in self.included_grids]
        grid_max_bounds = [grid.xc_bounds[1] for grid in self.included_grids]
        return [np.nanmin(grid_min_bounds), np.nanmax(grid_max_bounds)]

    @property
    def yc_bounds(self):
        grid_min_bounds = [grid.yc_bounds[0] for grid in self.included_grids]
        grid_max_bounds = [grid.yc_bounds[1] for grid in self.included_grids]
        return [np.nanmin(grid_min_bounds), np.nanmax(grid_max_bounds)]

    @property
    def included_grids(self):
        grids = []
        for i, grid in enumerate(self.grids):
            if i in self.ignore_list:
                continue
            grids.append(grid)
        return grids

    @property
    def times(self):
        return np.array([grid.reftime for grid in self.grids])

    @property
    def time_bounds(self):
        times = self.times
        return [np.nanmin(times), np.nanmax(times)]

    @property
    def n_grids(self):
        return len(self.grids)

    @property
    def proj4str(self):
        return self.grids[0].proj4str


class ALSL4Grid(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.offset = [0.0, 0.0]
        self._read_file()

    def _read_file(self):
        self.nc = xr.open_dataset(self.filepath, decode_times=False)

    @property
    def filename(self):
        return os.path.split(self.filepath)[-1]

    @property
    def reftime(self):
        time_value = self.nc.time.values[0]
        units = self.nc.time.units
        return num2date(time_value, units)

    @property
    def proj4str(self):
        proj_info = self.nc.Polar_Stereographic_Grid
        return proj_info.proj4_string

    @property
    def resolution(self):
        return float(self.nc.attrs["geospatial_lat_resolution"])

    @property
    def grid_xc_yc(self):
        xc, yc = self.nc.xc.values, self.nc.yc.values
        xc += self.offset[0]
        yc += self.offset[1]
        return np.meshgrid(xc, yc)

    @property
    def xc_bounds(self):
        xc = self.nc.xc.values
        xc += self.offset[0]
        return [np.nanmin(xc), np.nanmax(xc)]

    @property
    def yc_bounds(self):
        yc = self.nc.yc.values
        yc += self.offset[1]
        return [np.nanmin(yc), np.nanmax(yc)]

    @property
    def xcenter(self):
        return np.nanmean(self.xc_bounds)

    @property
    def ycenter(self):
        return np.nanmean(self.yc_bounds)

    @property
    def proj_extent(self):
        xc_bounds = self.xc_bounds
        yc_bounds = self.yc_bounds
        return [xc_bounds[0], yc_bounds[0], xc_bounds[1], yc_bounds[1]]

    @property
    def width(self):
        bounds = self.xc_bounds
        return bounds[1]-bounds[0]

    @property
    def height(self):
        bounds = self.yc_bounds
        return bounds[1]-bounds[0]

    @property
    def value(self):
        return self.nc.elevation.values

    @property
    def lons(self):
        return self.nc.lon.values

    @property
    def lats(self):
        return self.nc.lat.values


class ALSMergedGrid(object):

    def __init__(self, x_min, x_max, y_min, y_max, res_m, proj4str):
        """

        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :param res_m:
        :param proj4str:
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = res_m

        # Compute the shape of the full grid
        self.xc = np.linspace(self.x_min, self.x_max, (self.x_max-self.x_min) / res_m)
        self.yc = np.linspace(self.y_min, self.y_max, (self.y_max - self.y_min) / res_m)
        self.xy = np.meshgrid(self.xc, self.yc)
        self.dims = self.xy[0].shape
        self.grid = np.full(self.dims, np.nan)
        self.lons = np.full(self.dims, np.nan)
        self.lats = np.full(self.dims, np.nan)

        # Compute lon/lat of all grid cells
        self.proj4str = proj4str
        # logger.info("Compute lon/lat per grid cell (%s)" % proj4str)
        # p = pyproj.Proj(proj4str)
        # self.lon, self.lat = p(self.xy[0], self.xy[1], inverse=True)

    def add_grid(self, grid):

        # Compute the offset indices between merged grid and grid subset
        xi_offset = int((grid.xc_bounds[0]-self.xc_bounds[0])/self.res)
        yj_offset = int((grid.yc_bounds[0]-self.yc_bounds[0])/self.res)

        # Find finite values in the grid subset
        subset_valid_indices = np.where(np.isfinite(grid.value))

        subset_yj, subset_xi = subset_valid_indices[0].copy(), subset_valid_indices[1].copy()
        subset_yj += yj_offset
        subset_xi += xi_offset
        merged_valid_indices = (subset_yj, subset_xi)

        # TODO: Temporary fix to align grid segments, needs improvement on GPS solution
        self.grid[merged_valid_indices] = grid.value[subset_valid_indices]-np.nanmedian(grid.value)
        self.lons[merged_valid_indices] = grid.lons[subset_valid_indices]
        self.lats[merged_valid_indices] = grid.lats[subset_valid_indices]

    def export_netcdf(self, output_path):
        """
        Create a netcdf with the merged grid
        :param output_path:
        :return:
        """

        # Parameter
        grid_dims = ("yc", "xc")
        coord_dims = ("yc", "xc")

        # Collect all data vars
        data_vars = {"elevation": xr.Variable(grid_dims, self.grid.astype(np.float32)),
                     "lon": xr.Variable(coord_dims, self.lons.astype(np.float32)),
                     "lat": xr.Variable(coord_dims, self.lats.astype(np.float32))}

        # Add grid mapping
        # grid_mapping_name, grid_mapping_attrs = self.grid_mapping_items
        # if grid_mapping_name is not None:
        #     data_vars[grid_mapping_name] = xr.Variable(("grid_mapping"), [0], attrs=grid_mapping_attrs)

        # Collect all coords
        coords = {"xc": xr.Variable(("xc"), self.xc.astype(np.float32)),
                  "yc": xr.Variable(("yc"), self.yc.astype(np.float32))}

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # # Add global attributes
        # for key, value in self.dem.metadata.items:
        #     self.ds.attrs[key] = value

        # Turn on compression for all variables
        comp = dict(zlib=True)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)

    def export_geotiff(self, output_path):
        """
        Export a geotiff
        :param output_path:
        :return:
        """

        # Set the projection of the output dataset
        srs = osr.SpatialReference()
        srs.ImportFromProj4(self.proj4str)
        wkt = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, self.dims[1], self.dims[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform((self.x_min, self.res, 0, self.y_max, 0, -self.res))
        dataset.SetProjection(wkt)
        dataset.GetRasterBand(1).WriteArray(np.flipud(self.grid))
        dataset.GetRasterBand(1).SetNoDataValue(np.nan)
        dataset.FlushCache()  # Write to disk.

    @property
    def xc_bounds(self):
        return [np.nanmin(self.xc), np.nanmax(self.xc)]

    @property
    def yc_bounds(self):
        return [np.nanmin(self.yc), np.nanmax(self.yc)]

    @property
    def width(self):
        bounds = self.xc_bounds
        return bounds[1]-bounds[0]

    @property
    def height(self):
        bounds = self.yc_bounds
        return bounds[1]-bounds[0]