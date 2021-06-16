# -*- coding: utf-8 -*-

"""
A module containing the automated open water detection and conversion from elevation to freeboard
"""

__author__ = "Nils Hutter"


import numpy as np

from datetime import datetime, timedelta
import os
from loguru import logger
from scipy.signal import medfilt, convolve,find_peaks
from pathlib import Path
import matplotlib.pylab as plt
from collections import OrderedDict
import multiprocessing

from awi_als_toolbox.filter import ALSPointCloudFilter
from awi_als_toolbox.scripts import get_als_file, get_als_segments
from awi_als_toolbox import AlsDEMCfg


class DetectOpenWater(ALSPointCloudFilter):
    """
    Detects open water pixels using the elevation and refelctance data
    """

    def __init__(self,fov_resolution=0.05550000071525574,kernel_size=5,
                 rflc_thres=3.0,elev_tol=0.02,elev_segment=0.2,rflc_minmax=False,
                 export_file='open_water_points.csv'):
        """
        Initialize the filter.
        :param fov_resolution: Angular resolution of the field of view
        :param kernel_size: Kernel size of median filter to smooth elevation and reflectance data, also sets minimum peak width
        :param rflc_thres: Minimum differences of reflectance of a peak from the mean reflectance
        :param elev_tol: Elevation uncertainty for individual point measurements
        :param elev_segment: Allowed variation of sea surface height (incl. GPS height uncertainties) within a segment
        :param rflc_minmax: Use both minima and maxima (glint) in reflectance to detect open water (default: False)
        """
        super(DetectOpenWater, self).__init__(fov_resolution=fov_resolution,kernel_size=kernel_size,
                                              rflc_thres=rflc_thres,elev_tol=elev_tol,
                                              elev_segment=elev_segment,rflc_minmax=rflc_minmax,
                                              export_file=export_file)

    def apply(self, als, do_plot=False):
        """
        Apply the filter for all lines in the ALS data container
        :param als:
        :return:
        """

        logger.info("OpenWaterDetection is applied")
        
        # 1. Find NADIR pixels in data
        
        # Mask aircraft roll data for nans
        mask_roll = np.where(np.isfinite(als.get('aircraft_roll')))
        # Indexes of all NADIR pixels
        nadir_inds = (mask_roll[0],(np.ones(mask_roll[0].size)*als.n_shots/2+als.get('aircraft_roll')[mask_roll]/self.cfg["fov_resolution"]).astype('int'))
        # Subset NADIR elevation and reflectance data
        elev_nadir = als.get('elevation')[nadir_inds]
        rflc_nadir = als.get('reflectance')[nadir_inds]
        # Mask for invalid pixels
        mask = np.where(np.logical_and(np.isfinite(elev_nadir),np.isfinite(rflc_nadir)))

        
        # 2. Smoothen elevation and reflectance data for gridscale variations
        
        kernel_size = self.cfg["kernel_size"]
        elev_nadir_m = medfilt(elev_nadir[mask],kernel_size=kernel_size)
        rflc_nadir_m = medfilt(rflc_nadir[mask],kernel_size=kernel_size)
        
        
        # 3. Detect local minima in elevation and reflectance data
        
        # Miminal width of the peaks
        min_width = kernel_size
        # Peaks (minima) in elevation data (threshold used elevation<min(elevation)+elev_segment)
        elev_peaks_m, _ = find_peaks(-elev_nadir_m, height=0.9*np.nanmax(-elev_nadir_m)+0.1*np.median(-elev_nadir_m),width=min_width)
        # Peaks (minima) in reflectance data
        rflc_peaks_m, _ = find_peaks(-rflc_nadir_m,width=min_width)
        # Peaks (maxima) in reflectance data
        if self.cfg["rflc_minmax"]:
            logger.info(" - using also maxima of reflectance for open water detection")
            rflc_max_m, _ = find_peaks(rflc_nadir_m,width=min_width)

        
        # 4. Quality checks of detected peaks
        
        # Check for reflectance threshold
        rflc_thres = self.cfg["rflc_thres"]
        rflc_peaks_m = rflc_peaks_m[(np.mean(rflc_nadir_m)-rflc_nadir_m[rflc_peaks_m])>rflc_thres]
        if self.cfg["rflc_minmax"]:
            rflc_max_m = rflc_max_m[(rflc_nadir_m[rflc_max_m]-np.mean(rflc_nadir_m))>rflc_thres]
            # Combine filtered minima and maxima for further analyis
            rflc_peaks_m = np.concatenate([rflc_peaks_m,rflc_max_m])

        # Check for colocated peaks in elevation and reflectance
        if elev_peaks_m.size>0 and rflc_peaks_m.size>0:
            peaks_m = elev_peaks_m[[np.any(np.abs(rflc_peaks_m-ielev)<kernel_size/2) for ielev in elev_peaks_m]]
        else:
            peaks_m = np.array([])

        # Peaks in globale index
        peaks = mask[0][np.array([ipeak-int(kernel_size/2)+
                                  np.argmin(elev_nadir[mask][ipeak-int(kernel_size/2):ipeak+int(kernel_size/2)+1]) 
                                  for ipeak in peaks_m]).astype('int')]
        

        # Check for peaks relation to global minimum
        elev_tol = self.cfg["elev_tol"]
        elev_grad = self.cfg["elev_segment"]/als.n_lines
        globmin = np.nanmin(elev_nadir)
        iglobmin = np.where(elev_nadir==np.nanmin(elev_nadir))[0][0]
        peaks_glob = peaks[np.abs(elev_nadir[peaks]-globmin)<=np.abs(peaks-iglobmin)*elev_grad+elev_tol]
        
        
        # 5. Indexes of open water points
        open_water_inds = (nadir_inds[0][peaks_glob],nadir_inds[1][peaks_glob])
        logger.info(" - number of open water references detected: %i" %(peaks_glob.size))
        
        
        # 6. (optional) plot results of open water detection
        if do_plot:
            fig,ax = plt.subplots(2,2,sharex=True)

            ax[0,0].pcolormesh(als.get('elevation').T)
            ax[0,0].plot(nadir_inds[0],nadir_inds[1],'k--')
            ax[0,0].plot(nadir_inds[0][peaks],nadir_inds[1][peaks],'kx')
            ax[0,0].plot(nadir_inds[0][peaks_glob],nadir_inds[1][peaks_glob],'rx')
            ax[0,1].pcolormesh(als.get('reflectance').T)
            ax[0,1].plot(nadir_inds[0],nadir_inds[1],'k--')
            ax[0,1].plot(nadir_inds[0][peaks],nadir_inds[1][peaks],'kx')
            ax[0,1].plot(nadir_inds[0][peaks_glob],nadir_inds[1][peaks_glob],'rx')

            ax[1,0].plot(elev_nadir)
            ax[1,0].plot(peaks_glob, elev_nadir[peaks_glob], "+")

            ax[1,1].plot(rflc_nadir)
            ax[1,1].plot(peaks_glob, rflc_nadir[peaks_glob], "+")
         
        
        # 7. Export open water points
        self._export_open_water_points((nadir_inds[0][peaks_glob],nadir_inds[1][peaks_glob]),als)
        
  
    def _export_open_water_points(self, inds_peak, als):
        rflc_mean = np.nanmean(als.get('reflectance'))
        for ix,iy in zip(inds_peak[0],inds_peak[1]):
            with self._get_export() as export_file:
                export_file.write('%.15f,%.15f,%.15f,%f,%f,%f\n' %(als.get('timestamp')[ix,iy],
                                                             als.get('longitude')[ix,iy],
                                                             als.get('latitude')[ix,iy],
                                                             als.get('elevation')[ix,iy],
                                                             als.get('reflectance')[ix,iy],
                                                             als.get('reflectance')[ix,iy]-rflc_mean))
                export_file.close()
        
        
    def _get_export(self):
        # Check if file exists
        export_file = Path(self.cfg["export_file"]).absolute()
        if not export_file.is_file():
            # Otherwise create new file with header
            with export_file.open(mode='w') as f:
                f.write('timestamp,longitude,latitude,elevation,reflectance,reflectance (diff to mean)\n')
                f.close()
        # Open file to read
        return export_file.open(mode='a')
        

        
class AlsFreeboardConversion(object):
    """
    This class reads all binary ALS data (.alsbin2-files), detects open water, interpolates
    the sea surface heigth, and computes the snow freeboard from the elevation data.
    """
    
    def __init__(self, cfg=None):
        """
        Create a Freeboard Conversion object iwth cfg files
        :param cfg: dictionary containg the key arguments for OpenWaterDetection and SeaDurfaceInterpolation
        """
        self.cfg = cfg
        if cfg is None:
            self.cfg = OrderedDict()
            
        for ikey in ['OpenWaterDetection','SeaSurfaceInterpolation']:
            if ikey not in self.cfg.keys():
                self.cfg[ikey] = {}
                
    def open_water_detection(self, als_filepaths, dem_cfg, file_version=1,
                             ow_export_file=None, use_multiprocessing=False,mp_reserve_cpus=2):
        """
        Function that detects open water points in a list of .alsbin2-files. All open water points are 
        outputted to a common csv-file.
        :param als_filepaths: list of paths to .alsbin2-files to process
        :param ow_export_file: path to csv-file to export all open water points
        """
        
        # Generate common csv-file to output open water points
        if ow_export_file is not None:
            self.cfg['OpenWaterDetection']['export_file'] = ow_export_file
            
            with Path(self.cfg['OpenWaterDetection']["export_file"]).absolute().open(mode='w') as f:
                f.write('timestamp,longitude,latitude,elevation,reflectance,reflectance (diff to mean)\n')
                f.close()
                
        if 'export_file' in self.cfg['OpenWaterDetection'].keys():
            self.export_file = self.cfg['OpenWaterDetection']['export_file']
        else:
            self.export_file = Path('open_water_points.csv').absolute()
            
        
        # Get all segments from ALS files
        self.segments = get_als_segments(als_filepaths, dem_cfg, file_version=file_version)
        
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
            
        # Apply open water detection to all segements
        if use_multiprocessing:
            # Parallel processing of all segments
            iters = np.arange(len(self.segments['i']))
            np.random.shuffle(iters)
            results = [process_pool.apply_async(open_water_detection_wrapper, args=(self.segments['als_filepath'][i], 
                                                                                    dem_cfg, file_version,
                                                                                    self.segments['start_sec'][i], 
                                                                                    self.segments['stop_sec'][i], 
                                                                                    self.segments['i'][i], 
                                                                                    self.segments['n_segments'][i],
                                                                                    self.cfg)) 
                       for i in iters]
            result =[iresult.get() for iresult in results]
        else:
            # Loop over all segments
            for i in range(len(self.segments['i'])):
                open_water_detection_wrapper(self.segments['als_filepath'][i], dem_cfg, file_version, 
                                             self.segments['start_sec'][i], self.segments['stop_sec'][i], 
                                             self.segments['i'][i], self.segments['n_segments'][i], self.cfg)
        

        
def open_water_detection_wrapper(als_filepath, dem_cfg, file_version, start_sec, stop_sec, i, n_segments, cfg):
    # Read ALS Data
    alsfile = get_als_file(Path(als_filepath), file_version, dem_cfg)
    
    logger.info("Processing %s: [%g:%g] (%g/%g)" % (str(als_filepath), start_sec, stop_sec, i+1, n_segments))
    als = alsfile.get_data(start_sec, stop_sec)
    
    # Initiate Open water detection object
    owfilter = DetectOpenWater(**cfg['OpenWaterDetection'])
    
    # detect open water
    owfilter.apply(als)
    