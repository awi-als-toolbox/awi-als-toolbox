# -*- coding: utf-8 -*-

"""
Module that contains functions for standardized ALS processing workflows. These are
meant to be called by more specific scripts.
"""

import sys
import multiprocessing
from pathlib import Path

# This matplotlib setting is necessary if the script
# is run in a shell via ssh and no window manager
# import matplotlib
# matplotlib.use("agg")

from loguru import logger

from . import AirborneLaserScannerFile, AirborneLaserScannerFileV2, AlsDEM
from .export import AlsDEMNetCDF


def als_l1b2dem(als_filepath, dem_cfg, output_cfg, file_version=1, use_multiprocessing=False,
                mp_reserve_cpus=2):
    """
    Grid a binary point cloud file with given grid specification and in segments of
    a given temporal coverage
    :param als_filepath: (str, pathlib.Path): The full filepath of the binary ALS point cloud file
    :param dem_cfg: (awi_als_toolbox.demgen.AlsDEMCfg):
    :param output_cfg:
    :param file_version:
    :param use_multiprocessing:
    :param mp_reserve_cpus:
    :return:
    """

    # --- Step 1: connect to the ALS binary point cloud file ---
    #
    # At the moment there are two options:
    #
    #   1) The binary point cloud data from the "als_level1b" IDL project.
    #      The output is designated as file version 1
    #
    #   2) The binary point cloud data from the "als_level1b_seaice" IDL project.
    #      The output is designated as file version 2 and can be identified
    #      by the .alsbin2 file extension

    # Input validation
    als_filepath = Path(als_filepath)
    if not als_filepath.is_file():
        logger.error("File does not exist: {}".format(str(als_filepath)))
        sys.exit(1)

    # Connect to the input file
    # NOTE: This step will not read the data, but read the header metadata information
    #       and open the file for sequential reading.
    logger.info("Open ALS binary file: {} (file version: {})".format(als_filepath.name, file_version))
    if file_version == 1:
        alsfile = AirborneLaserScannerFile(als_filepath, **dem_cfg.connect_keyw)
    elif file_version == 2:
        alsfile = AirborneLaserScannerFileV2(als_filepath)
    else:
        logger.error("Unknown file format: {}".format(dem_cfg.input.file_version))
        sys.exit(1)

    # --- Step 3: loop over the defined segments ---
    # Get a segment list based on the suggested segment lengths for the gridding preset
    # TODO: Evaluate the use of multi-processing for the individual segments.
    segments = alsfile.get_segment_list(dem_cfg.segment_len_secs)
    n_segments = len(segments)
    logger.info("Split file in %d segments" % n_segments)

    # Substep (Only valid if multi-processing should be used
    process_pool = None
    if use_multiprocessing:
        # Estimate how much workers can be added to the pool
        # without overloading the CPU
        n_processes = multiprocessing.cpu_count()
        n_processes -= mp_reserve_cpus
        n_processes = n_processes if n_processes > 1 else 1
        # Create process pool
        logger.info("Use multi-processing with {} workers".format(n_processes))
        process_pool = multiprocessing.Pool(n_processes)

    for i, (start_sec, stop_sec) in enumerate(segments):

        # Extract the segment
        # NOTE: This includes file I/O
        logger.info("Processing %s [%g:%g] (%g/%g)" % (als_filepath.name, start_sec, stop_sec, i+1, n_segments))
        als = alsfile.get_data(start_sec, stop_sec)

        # TODO: Replace with try/except with actual Exception
        # except BaseException:
        #     msg = "Unhandled exception while reading %s:%g-%g -> Skip segment"
        #     logger.error(msg % (als_filepath.name, start_sec, stop_sec))
        #     print(sys.exc_info()[1])
        #     continue

        # Apply any filter defined
        for input_filter in dem_cfg.get_input_filter():
            input_filter.apply(als)

        # Validate segment
        # -> Do not try to grid a segment that has no valid elevations
        if not als.has_valid_data:
            msg = "... No valid data in {}:{}-{} -> skipping segment"
            msg = msg.format(als_filepath.name, start_sec, stop_sec)
            logger.warning(msg)
            continue

        # Grid the data and write the output in a netCDF file
        # This can either be run in parallel or
        if use_multiprocessing:
            result = process_pool.apply_async(gridding_workflow, args=(als, dem_cfg, output_cfg, ))
            # print(result.get())
        else:
            gridding_workflow(als, dem_cfg, output_cfg)

    if use_multiprocessing:
        process_pool.close()
        process_pool.join()


def gridding_workflow(als, dem_cfg, output_cfg):
    """
    Single function gridding and plot creation that can be passed to a multiprocessing process
    :param als: (ALSData) ALS point cloud data
    :param dem_cfg:
    :param output_cfg:
    :return: None
    """

    # Grid the data
    logger.info("... Start gridding")

    dem = AlsDEM(als, cfg=dem_cfg)
    dem.create()

    # try:
    #     dem = AlsDEM(als, cfg=dem_cfg)
    #     dem.create()
    # except:
    #     logger.error("Unhandled exception while gridding -> skip gridding")
    #     print(sys.exc_info()[1])
    #     return
    logger.info("... done")

    # create
    nc = AlsDEMNetCDF(dem, output_cfg)
    nc.export()
    logger.info("... exported to: %s" % nc.path)
