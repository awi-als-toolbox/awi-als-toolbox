# -*- coding: utf-8 -*-

"""
Module that contains functions for standardized ALS processing workflows. These are
meant to be called by more specific scripts.
"""

import os
import sys

# This matplotlib setting is necessary if the script
# is run in a shell via ssh and no window manager
import matplotlib
# matplotlib.use("agg")

from loguru import logger

from awi_als_toolbox.data import FlightGPSData
from awi_als_toolbox.reader import AirborneLaserScannerFile
from awi_als_toolbox.demgen import AlsDEM, AlsDEMCfg
from awi_als_toolbox.export import AlsDEMNetCDF
from awi_als_toolbox.filter import AtmosphericBackscatterFilter


def als_l1b_dem_generation_workflow(als_filepath, grid_preset, parameter, dem_cfg,
                                    gps=None, metadata=None, **connect_keyw):
    """
    Creates quickview plots of a specific ALS l1b elevation file
    :param source_dir: (str) the path of the laserscanner file
    :param als_filename: (dict) configuration including the filename and the preset for gridding/quickview process
    :param grid_preset: (str) name of the gridding preset (sea_ice_low or sea_ice_high)
    :param gps: (xarray.Dataset) gps data of the entire flight
    :param metadata: (dict) metadata dictionary
    :param connect_keyw: (dict) keywords to be passed for alsfile.connect (e.g. device_name_override)
    :return: None
    """

    # Step 1: connect to the laserscanner file
    source_dir, als_filename = os.path.split(als_filepath)
    logger.info("Open ALS binary file: %s" % als_filename)
    try:
        alsfile = AirborneLaserScannerFile(als_filepath, **connect_keyw)
    except BaseException:
        logger.error("Unexpected error -> skip file")
        print(sys.exc_info()[1])
        return

    # Get the gridding settings
    dem_cfg = AlsDEMCfg.preset(grid_preset, **dem_cfg)

    # Get a segment list based on the suggested segment lengths for the gridding preset
    segments = alsfile.get_segment_list(dem_cfg.segment_len_secs)
    n_segments = len(segments)

    flightdata = None
    if gps is not None:
        logger.info("Adding GPS flight data")
        seconds = gps.TIME.values.astype(float)*0.001
        time = alsfile.timestamp2time(seconds)
        flightdata = FlightGPSData(time, gps.LONGITUDE.values, gps.LATITUDE.values, gps.ALTITUDE.values)

    # Only necessary if multiprocessing is used
    logger.info("Split file in %d segments" % n_segments)
    for i, (start_sec, stop_sec) in enumerate(segments):

        logger.info("Processing %s [%g:%g] (%g/%g)" % (als_filename, start_sec, stop_sec, i+1, n_segments))

        # Extract the segment
        try:
            als = alsfile.get_data(start_sec, stop_sec)
        except BaseException:
            msg = "Unhandled exception while reading %s:%g-%g -> Skip segment"
            logger.error(msg % (als_filename, start_sec, stop_sec))
            print(sys.exc_info()[1])
            continue

        # Apply atmospheric filter
        atmfilter = AtmosphericBackscatterFilter()
        atmfilter.apply(als)

        if metadata is not None:
            logger.info("Adding metadata")
            als.metadata.set_attributes(metadata["global_attrs"])
            als.metadata.set_variable_attributes(metadata["variable_attrs"])

        # Validate segment
        # -> Do not try to grid a segment that has no valid elevations
        if not als.has_valid_data:
            logger.error("... Invalid data in %s:%g-%g -> skipping segment" % (als_filename, start_sec, stop_sec))
            continue

        if flightdata is not None:
            als.set_flightdata(flightdata)

        # Grid the data and create a netCDF
        # NOTE: This can be run in parallel for different segments, therefore option to use
        #       multiprocessing
        export_dir = source_dir
        gridding_workflow(als, dem_cfg, export_dir)


def gridding_workflow(als, dem_cfg, export_dir):
    """
    Single function gridding and plot creation that can be passed to a multiprocessing process
    :param als: (ALSData) ALS point cloud data
    :param dem_cfg: (dict) DEM generation settings
    :param export_dir: (str) the target directory for the gridded netcdfs
    :return: None
    """

    # Grid the data
    logger.info("... Start gridding")
    try:
        dem = AlsDEM(als, cfg=dem_cfg)
        dem.create()
    except:
        logger.error("Unhandled exception while gridding -> skip gridding")
        print(sys.exc_info()[1])
        return
    logger.info("... done")

    # Create the quickview plot and save as png
    nc = AlsDEMNetCDF(dem, export_dir, project="mosaic", parameter="elevation")
    nc.export()
    logger.info("... exported to: %s" % nc.path)
