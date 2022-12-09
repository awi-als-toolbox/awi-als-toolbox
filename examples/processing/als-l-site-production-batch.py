
# -*- coding: utf-8 -*-

"""
This script can be used to create gridded segments from point cloud data
of multiple ALS L1b/L2 file in AWI binary format
"""

import os

from loguru import logger
from datetime import datetime
import numpy as np

import argparse
from pathlib import Path

#import psutil

from awi_als_toolbox import AlsDEMCfg, ALSGridCollection, get_yaml_cfg
from awi_als_toolbox.scripts import *
from awi_als_toolbox.export import AlsDEMNetCDFCfg
from awi_als_toolbox.freeboard import AlsFreeboardConversion


def main():
    """
    This script executes the gridding of ALS point cloud data based using
    the awi-als-toolbox package.

    Input arguments are to be supplied via command line arguments
    """

    # Get command line arguments and validate input data types
    args = ScriptArguments()

    # 1.) Grid binary data into 30s segment netcdf files
     
    # Initialize the DEM generation configuration object
    dem_cfg = AlsDEMCfg.from_cfg(args.result.dem_cfg)
    
    # Check for freeboard conversion
    if 'freeboard' in get_yaml_cfg(args.result.output_cfg)['variable_attributes'].keys():
        compute_freeboard = True
    else:
        compute_freeboard = False
        
        
    # Processing chain for L-site triangles and all other transect flights
    # 1. Run open water detection
    # 2. Grid all files with:
    #     - atmospheric backscatter filter
    #     - freeboard conversion
    # 
    # Difference to floe grids:
    #  - no ice drift correction is applied
    #  - no elevation correction is applied prior to open water detection
    
        
    # Initialize the output
    # Use export directory from arguments if present
    # (Else, use the directory of the input file)
    export_dir = args.result.export_dir
    os.umask(0o002)
    if export_dir is None:
        # Creates two different output directories: one for gridded not offset corrected files
        # and another one for gridded files that are offset corrected and include freeboard
        export_dir = args.result.als_l1b_dirpath.parent.joinpath('l4/icecs/')
    if not export_dir.parent.is_dir():
        os.mkdir(export_dir.parent)
    if not export_dir.is_dir():
        os.mkdir(export_dir)
    output_cfg = AlsDEMNetCDFCfg.from_cfg(args.result.output_cfg, export_dir=export_dir)
        
    
    # Read the gps data for the full flight
    als_filepaths = [os.path.join(args.result.als_l1b_dirpath,ifile) for ifile in 
                     os.listdir(args.result.als_l1b_dirpath) 
                     if ifile.endswith('.alsbin2')]
    als_filepaths.sort()
    print(als_filepaths)
    
    # Freeboard conversion
    if compute_freeboard:
        logger.info('Freeboard conversion detected: Start open water point detection')
        # Set output file for open water points
        try:
            ow_export_file = Path(dem_cfg.freeboard['OpenWaterDetection']['export_file']).absolute()
        except:
            # Set open water csv file
            ow_export_file = Path(export_dir).joinpath('open_water_points.csv')
            if dem_cfg.freeboard is None:
                dem_cfg.freeboard = {}
            if 'OpenWaterDetection' not in dem_cfg.freeboard.keys():
                dem_cfg.freeboard['OpenWaterDetection'] = {}
            dem_cfg.freeboard['OpenWaterDetection']['export_file'] = ow_export_file

        if args.result.skip_open_water_detection==0:
            # Initialise Freeboard conversion object
            ALSfreeboard = AlsFreeboardConversion(cfg=dem_cfg.freeboard)
            # Run open water detection
            ALSfreeboard.open_water_detection(als_filepaths, dem_cfg, file_version=2,
                                              use_multiprocessing=True)
                
    # Add reference time of overall flight for Ice Drift Correction
    #print(als_filepaths[0])
    reftimes = [datetime.strptime(als_filepaths[0].split('/')[-1].split('_')[-2],'%Y%m%dT%H%M%S'),
                datetime.strptime(als_filepaths[-1].split('/')[-1].split('_')[-1].split('.')[0],'%Y%m%dT%H%M%S')]
    for ifilter in dem_cfg.input_filter:
        if ifilter['pyclass'] == 'IceDriftCorrection':
            ifilter['keyw']['reftimes'] = reftimes

    # Check for already processesed files
    proc_files = [ifile for ifile in os.listdir(export_dir) if ifile.endswith('-fv2p0.nc')]
    if len(proc_files)>0:
        proc_files.sort()
        proc_start = datetime.strptime(proc_files[0].split('-')[-3],'%Y%m%dT%H%M%S')
        proc_end   = datetime.strptime(proc_files[-1].split('-')[-2],'%Y%m%dT%H%M%S')
    
    # Grid ALS data
    if args.result.skip_grid==0: #or iround!=0:
        logger.info("Batch process: %d files" % len(als_filepaths))
        for als_filepath in als_filepaths:
            if len(proc_files)<0 or args.result.cont_proc==0:
                processed = False
            else:
                # Check if dates have already been processed
                alsfile_dates = [datetime.strptime(als_filepath.split('/')[-1].split('_')[-2],'%Y%m%dT%H%M%S'),
                                 datetime.strptime(als_filepath.split('/')[-1].split('_')[-1].split('.')[0],'%Y%m%dT%H%M%S')]
                processed = np.all([(alsfile_dates[0]-proc_start).total_seconds()>=-1,
                                    (alsfile_dates[-1]-proc_end).total_seconds()<=1])
            if processed:
                logger.info("Skip processing of file %s as processed version exists already" % als_filepath)
            else:
                logger.info("Start processing with file process: %d files" % len(als_filepaths))
                als_l1b2dem(als_filepath, dem_cfg, output_cfg, file_version=args.result.file_version,
                            use_multiprocessing=True)
            
                
    # Finish statement processing
    logger.info('Status: DONE - l4 processing finished successfully')
    

        
        
class ScriptArguments(object):
    """
    Container for input script arguments
    """

    def __init__(self):
        """
        Reads and evalulations command line arguments to this script
        """

        def is_valid_file(parser, arg):
            arg = Path(arg).absolute()
            if not arg.is_file():
                parser.error("The file {} does not exist!".format(arg))
            else:
                return arg
        
        def is_valid_dir(parser, arg):
            arg = Path(arg).absolute()
            if not arg.is_dir():
                parser.error("The directory %s does not exist!".format(arg))
            else:
                return arg

        # Create the parser
        self.parser = argparse.ArgumentParser(description="ALS gridding workflow")

        self.parser.add_argument("als_l1b_dirpath", type=lambda x: is_valid_dir(self.parser, x),
                                 help='ALS l1b file')
        
        self.parser.add_argument("--output-cfg", type=lambda x: is_valid_file(self.parser, x),
                                 help='netCDF metdata definition file (*.yaml)',
                                 required=True, action="store", dest="output_cfg")
        
        self.parser.add_argument("--dem-cfg", type=lambda x: is_valid_file(self.parser, x),
                                 help='DEM generation config file (*.yaml)',
                                 required=True, action="store", dest="dem_cfg")

        self.parser.add_argument("--file-version", type=int, default=1,
                                 help='File version of the ALS binary file (1: pre-2021, 2: for *.alsbin2 file',
                                 choices=[1, 2], required=False, action="store", dest="file_version")
        
        self.parser.add_argument("--skip-grid", type=int, default=0,
                                 help='skip gridding in 30s segments',
                                 choices=[0, 1], required=False, action="store", dest="skip_grid")
        
        self.parser.add_argument("--continue-processing", type=int, default=0,
                                 help='skip processing of 30s segments that are already in export dir',
                                 choices=[0, 1], required=False, action="store", dest="cont_proc")
        
        self.parser.add_argument("--skip-open-water-detection", type=int, default=0,
                                 help='skip detecting open water points',
                                 choices=[0, 1], required=False, action="store", dest="skip_open_water_detection")
        
        self.parser.add_argument("--freeboard-iter", type=int, default=0,
                                 help='Which iteration is computed: 0 - gridding for offset correction, 1 - freeboard conversion and gridding',
                                 choices=[0, 1], required=False, action="store", dest="freeboard_iter")

        self.parser.add_argument("--export-dir", type=str, default=None,
                                 help='Target directory for the netCDF files',
                                 required=False, action="store", dest="export_dir")

        # Parse the command line arguments
        self.result = self.parser.parse_args()


        
if __name__ == "__main__":
    main()
