# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import numpy as np
import xarray as xr
from pathlib import Path
from collections import OrderedDict

from ._utils import get_yaml_cfg


class AlsDEMNetCDFCfg(object):
    """
    Container for the netCDF output structure
    """

    def __init__(self, filenaming, global_attributes, variable_attributes, export_dir=None, correcting_fields=[]):
        """

        :param filenaming:
        :param gattrs:
        :param vars:
        """
        self.filenaming = filenaming
        self.global_attributes = global_attributes
        self.variable_attributes = variable_attributes
        self.export_dir = export_dir
        self.correcting_fields = correcting_fields

    @classmethod
    def from_cfg(cls, yaml_filepath, **kwargs):
        """
        Initialize the class from a yaml config file
        :param yaml_filepath:
        :return:
        """
        cfg = get_yaml_cfg(yaml_filepath)
        cfg.update(**kwargs)
        return cls(**cfg)

    def get_var_attrs(self, variable_name):
        """
        Get the variable attributes for a given variable name
        :param variable_name:
        :return: dict or None
        """
        return self.variable_attributes.get(variable_name, None)


class AlsDEMNetCDF(object):

    def __init__(self, dem, cfg):
        """

        :param dem:
        """
        self.dem = dem
        self.export_dir = cfg.export_dir
        self.cfg = cfg
        # self.project = project
        # self.parameter = parameter
        # self.export_dir = export_dir
        #
        # if filename == "auto":
        #     template = "awi-{project}-{proc_level}-{parameter}-vq580-stere_{res}-{tcs}-{tce}-fv1p0.nc"
        #     self.filename = template.format(proc_level=self.dem.fn_proc_level, res=self.dem.fn_res,
        #                                     project=self.project, parameter=self.parameter,
        #                                     tcs=self.dem.fn_tcs, tce=self.dem.fn_tce)
        # else:
        #     self.filename = filename

        # Construct the dataset
        # NOTE: The actual export procedure is handled by the export method to allow custom modification
        self._construct_xr_dataset()

    def _construct_xr_dataset(self):
        """
        Create a xarray.Dataset instance of the DEM
        :return:
        """

        # Parameter
        grid_dims = coord_dims = ("yc", "xc")

        # Collect all gridded parameter
        data_vars = OrderedDict()
        output_variable_names = self.cfg.variable_attributes.keys()
        for grid_variable_name in self.dem.grid_variable_names:
            if grid_variable_name not in output_variable_names:
                continue
            var = self.dem.get_variable(grid_variable_name, masked=True)
            xrvar = xr.Variable(grid_dims, var, attrs=self.cfg.get_var_attrs(grid_variable_name))
            data_vars[grid_variable_name] = xrvar

        # Add additional variables
        data_vars["n_points"] = xr.Variable(grid_dims, self.dem.n_shots.astype(np.int16),
                                            attrs=self.cfg.get_var_attrs("n_points"))
        data_vars["lon"] = xr.Variable(coord_dims, self.dem.lon.astype(np.float32),
                                       attrs=self.cfg.get_var_attrs("lon"))
        data_vars["lat"] = xr.Variable(coord_dims, self.dem.lat.astype(np.float32),
                                       attrs=self.cfg.get_var_attrs("lat"))

        # Add grid mapping
        grid_mapping_name, grid_mapping_attrs = self.dem.grid_mapping_items
        if grid_mapping_name is not None:
            data_vars[grid_mapping_name] = xr.Variable("grid_mapping", [0], attrs=grid_mapping_attrs)

        # Get the dimension variables
        coords = {"time": xr.Variable("time", [self.dem.ref_time], attrs=self.cfg.get_var_attrs("time")),
                  "time_bnds": xr.Variable("time_bnds", self.dem.time_bnds, attrs=self.cfg.get_var_attrs("time_bnds")),
                  "xc": xr.Variable("xc", self.dem.xc.astype(np.float32), attrs=self.cfg.get_var_attrs("xc")),
                  "yc": xr.Variable("yc", self.dem.yc.astype(np.float32), attrs=self.cfg.get_var_attrs("yc"))}

        self.ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add global attributes
        for key, value in self.dem.metadata.items:
            self.ds.attrs[key] = value

    def export(self):
        """
        Export the grid data as netcdf via xarray.Dataset
        :param filename:
        :return:
        """
        # Turn on compression for all variables
        comp = dict(zlib=True)
        encoding = {var: comp for var in self.ds.data_vars}
        self.ds.to_netcdf(self.path, engine="netcdf4", encoding=encoding)

    @property
    def filename(self):
        """
        Construct the filename
        TODO:
        :return:
        """
        template = str(self.cfg.filenaming)
        filename = template.format(proc_level=self.dem.fn_proc_level, res=self.dem.fn_res,
                                   tcs=self.dem.fn_tcs, tce=self.dem.fn_tce)
        return filename

    @property
    def path(self):
        return Path(self.export_dir) / self.filename
