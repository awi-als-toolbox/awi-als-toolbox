# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import os
import numpy as np
import xarray as xr


class AlsDEMNetCDF(object):

    def __init__(self, dem, export_dir, filename="auto"):
        """

        :param dem:
        """
        self.dem = dem
        self.export_dir = export_dir

        if filename == "auto":
            template = "awi-icebird-{proc_level}-freeboard-vq580-stere_{res}-{tcs}-{tce}-fv1p0.nc"
            self.filename = template.format(proc_level=self.dem.fn_proc_level, res=self.dem.fn_res,
                                            tcs=self.dem.fn_tcs, tce=self.dem.fn_tce)
        else:
            self.filename = filename

        # Construct the dataset
        # NOTE: The actual export procedure is handled by the export method to allow custom modification
        self._construct_xr_dataset()

    def _construct_xr_dataset(self):
        """
        Create a xarray.Dataset instance of the DEM
        TODO: This could be moved in the demgen module
        :return:
        """

        # Parameter
        grid_dims = ("yc", "xc")
        coord_dims = ("yc", "xc")
        metadata = self.dem.metadata

        # Collect all data vars
        data_vars = {"freeboard": xr.Variable(grid_dims, self.dem.dem_z.astype(np.float32),
                                              attrs=metadata.get_var_attrs("freeboard")),
                     "n_points": xr.Variable(grid_dims, self.dem.n_shots.astype(np.int16),
                                             attrs=metadata.get_var_attrs("n_points")),
                     "lon": xr.Variable(coord_dims, self.dem.lon.astype(np.float32),
                                        attrs=metadata.get_var_attrs("lon")),
                     "lat": xr.Variable(coord_dims, self.dem.lat.astype(np.float32),
                                        attrs=metadata.get_var_attrs("lat"))}

        # Collect all coords
        coords = {"xc": xr.Variable(("xc"), self.dem.xc.astype(np.float32),
                                    attrs=metadata.get_var_attrs("xc")),
                  "yc": xr.Variable(("yc"), self.dem.yc.astype(np.float32),
                                    attrs=metadata.get_var_attrs("yc"))}

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
    def path(self):
        return os.path.join(self.export_dir, self.filename)