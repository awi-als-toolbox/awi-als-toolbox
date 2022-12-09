
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
        
        
    # NEW ORDER
    # Round 1:
    #   1. set up export dir to /icecs/
    #   2. grid all .alsbin2 files (without freeboard)
    #   3. merge grid and compute offset correction terms
    # Round 2:
    #   1. Set up new export dir
    #   2. deactivate offset correction computation
    #   3. detect open water -> but with offset_correction ALSPointCloudFilter (needs to be implemented!!)
    #   4. grid all .alsbin2 files
    #   5. merge all grids
    #   6. export floe grids
    
    for iround in [args.result.freeboard_iter]:#range(1+int(compute_freeboard)):
        roundstr = ['FIRST ROUND OF GRIDING AND MERGING TO COMPUTE OFFSET CORRECTION',
                    'SECOND ROUND OF GRIDING TO COMPUTE FREEBOARD AND MERGE FINAL FLOE GRIDS']
        logger.info(roundstr[iround])
        
        # NEED TO CHANGE IN BOTH ITERATION
        # Initialize the output
        # Use export directory from arguments if present
        # (Else, use the directory of the input file)
        export_dir = args.result.export_dir
        os.umask(0o002)
        if export_dir is None:
            # Creates two different output directories: one for gridded not offset corrected files
            # and another one for gridded files that are offset corrected and include freeboard
            export_dir = args.result.als_l1b_dirpath.parent.joinpath('l4/icecs%s/' %['','_freeboard'][iround])
            if not export_dir.parent.is_dir():
                os.mkdir(export_dir.parent)
            if not export_dir.is_dir():
                os.mkdir(export_dir)
        output_cfg = AlsDEMNetCDFCfg.from_cfg(args.result.output_cfg, export_dir=export_dir)
        
        # REMOVE FREEBOARD VARIABLE IN FIRST ITERATION
        if iround==0 and compute_freeboard:
            freeboard_attributes = output_cfg.variable_attributes['freeboard']
            freeboard_unc_attributes = output_cfg.variable_attributes['freeboard_uncertainty']
            ssh_attributes = output_cfg.variable_attributes['sea_surface_height']
            del output_cfg.variable_attributes['freeboard']
            del output_cfg.variable_attributes['freeboard_uncertainty']
            del output_cfg.variable_attributes['sea_surface_height']
        #elif iround==1:
        #    output_cfg.variable_attributes['freeboard'] = freeboard_attributes
        #    output_cfg.variable_attributes['freeboard_uncertainty'] = freeboard_unc_attributes
        #    output_cfg.variable_attributes['sea_surface_height'] = ssh_attributes
    
    
        # SAME IN BOTH ITERATIONS
        # Read the gps data for the full flight
        als_filepaths = [os.path.join(args.result.als_l1b_dirpath,ifile) for ifile in 
                         os.listdir(args.result.als_l1b_dirpath) 
                         if ifile.endswith('.alsbin2')]
        als_filepaths.sort()
        
        # Add reference time of overall flight for Ice Drift Correction
        #   (needs to be done berfore open water correction to use the same filter inputs
        #   in the open water detection as in the gridding.)
        reftimes = [datetime.strptime(als_filepaths[0].split('/')[-1].split('_')[-2],'%Y%m%dT%H%M%S'),
                    datetime.strptime(als_filepaths[-1].split('/')[-1].split('_')[-1].split('.')[0],'%Y%m%dT%H%M%S')]
        for ifilter in dem_cfg.input_filter:
            if ifilter['pyclass'] == 'IceDriftCorrection':
                ifilter['keyw']['reftimes'] = reftimes
                ifilter['keyw']['reftime']  = reftimes[0]+0.5*(reftimes[1]-reftimes[0])
                
    
        # ONLY IN SECOND ITERATION
        if iround>0:
            # Check for freeboard conversion
            # if compute_freeboard:
            logger.info('FBCONV: Freeboard conversion detected')
            
            # Load previously computed elevation correction
            export_dir0 = args.result.als_l1b_dirpath.parent.joinpath('l4/icecs/grids/')
            corfiles = [ifile for ifile in os.listdir(export_dir0) if ifile.endswith('_correction.csv')]
            for ifile in corfiles:
                if not Path('./').joinpath(ifile).is_file():
                    logger.info('ELEVCOR: Linked and loaded computed elevation correction terms from export_dir: %s' %ifile)
                    os.symlink(Path(export_dir0).joinpath(ifile),Path('./').joinpath(ifile))
            logger.info('%s' %os.listdir())
            logger.info('ELEVCOR: deactivated new computation of offset correction terms')
            
            
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
                dem_cfg.freeboard['OpenWaterDetection']['floe_grid'] = True

            if args.result.skip_open_water_detection==0:
                logger.info('FBCONV: Freeboard conversion detected: Start open water point detection')
                # Initialise Freeboard conversion object
                ALSfreeboard = AlsFreeboardConversion(cfg=dem_cfg.freeboard)
                # Run open water detection
                ALSfreeboard.open_water_detection(als_filepaths, dem_cfg, file_version=2,
                                                  use_multiprocessing=True)
                
                
        # IN BOTH ITERATIONS
        # Grid ALS data
        if args.result.skip_grid==0: #or iround!=0:
            logger.info("Batch process: %d files" % len(als_filepaths))
            for als_filepath in als_filepaths:
                logger.info("Start processing with file process: %d files" % len(als_filepaths))
                als_l1b2dem(als_filepath, dem_cfg, output_cfg, file_version=args.result.file_version,
                            use_multiprocessing=True)
    
        # 2.) Merge all segment data to one floe grid and export

        # List all segment data and add to collection
        als_path = export_dir
        files = [als_path.joinpath(ifile).absolute() for ifile in os.listdir(als_path.absolute()) if ifile.endswith('.nc')]
        files.sort()
        l4_collect = ALSGridCollection(files, res=0.5)

        # Read config file and define export dir
        cfg = AlsDEMNetCDFCfg.from_cfg(args.result.grid_output_cfg)
        cfg.offset_correction = dem_cfg.offset_correction
        cfg.ice_drift_correction = dem_cfg.input_filter[np.where([ifilt['pyclass']=='IceDriftCorrection' 
                                                                  for ifilt in dem_cfg.input_filter])[0][0]]
        if cfg.export_dir is None:
            export_dir = Path(export_dir).joinpath('grids')
            cfg.export_dir = export_dir
            if not export_dir.is_dir():
                os.mkdir(export_dir)
        # REMOVE FREEBOARD VARIABLE IN FIRST ITERATION
        #REMOVE OFFSET CORRECTION IN SECOND ITERATION
        if iround==0 and compute_freeboard:
            freeboard_attributes_fg = cfg.variable_attributes['freeboard']
            freeboard_unc_attributes_fg = cfg.variable_attributes['freeboard_uncertainty']
            ssh_attributes_fg = cfg.variable_attributes['sea_surface_height']
            del cfg.variable_attributes['freeboard']
            del cfg.variable_attributes['freeboard_uncertainty']
            del cfg.variable_attributes['sea_surface_height']
        elif iround==1:
            #cfg.variable_attributes['freeboard'] = freeboard_attributes_fg
            cfg.offset_correction['correcting_fields'] = []

        # Read all segments and merge to grid
        ## - Use low reflectance tie points (points that corrected to the elevation reference height -> most likely leads and open water)
        ##   only in winter month prior to ice melt to avoid correcting of melt ponds
        #use_lrflt = np.any([np.all([idate.month<5 for idate in reftimes]), # earlier than May
        #                    np.all([idate.month>9 for idate in reftimes])]) # later than September
        #logger.info('Usage of low reflectance tie points is %s' %['DEACTIVATED (potential melt season)','ACTIVATED (winter)'][use_lrflt])
        
        floegrid = l4_collect.get_merged_grid(return_fnames=False,cfg=cfg)#,use_low_reflectance_tiepoints=use_lrflt)
    
        # ONLY IN SECOND ITERATION
        if True:# iround==1 or compute_freeboard==False:
            # Export netcdf file
            floegrid.export_netcdf()

            # Export geotiffs
            floegrid.export_geotiff()
        
        #print(psutil.virtual_memory())
        #del floegrid
        #print(psutil.virtual_memory())
        
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
        
        self.parser.add_argument("--grid-output-cfg", type=lambda x: is_valid_file(self.parser, x),
                                 help='netCDF metdata definition file for floegrid (*.yaml)',
                                 required=True, action="store", dest="grid_output_cfg")

        self.parser.add_argument("--dem-cfg", type=lambda x: is_valid_file(self.parser, x),
                                 help='DEM generation config file (*.yaml)',
                                 required=True, action="store", dest="dem_cfg")

        self.parser.add_argument("--file-version", type=int, default=1,
                                 help='File version of the ALS binary file (1: pre-2021, 2: for *.alsbin2 file',
                                 choices=[1, 2], required=False, action="store", dest="file_version")
        
        self.parser.add_argument("--skip-grid", type=int, default=0,
                                 help='skip gridding in 30s segments',
                                 choices=[0, 1], required=False, action="store", dest="skip_grid")
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
