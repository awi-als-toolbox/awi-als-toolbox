# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import os
import pyproj
import xarray as xr
from netCDF4 import num2date, date2num

import numpy as np

from osgeo import gdal, osr

from loguru import logger

from awi_als_toolbox.utils import geo_inverse


class ALSGridCollection(object):

    def __init__(self, filepaths, res=None, ignore_list=[]):
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

        # TODO: Temporary fix to align grid segements, needs improvement on GPS solution
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
        coords = {#"time": xr.Variable(("time"), [self.ref_time]),
                  "xc": xr.Variable(("xc"), self.xc.astype(np.float32)),
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