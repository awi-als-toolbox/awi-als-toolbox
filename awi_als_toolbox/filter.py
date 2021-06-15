# -*- coding: utf-8 -*-

"""
A module containing filter algorithm for the ALS point cloud data.
"""

__author__ = "Stefan Hendricks"


import numpy as np

import floenavi
from floenavi.polarstern import PolarsternAWIDashboardPos
from icedrift import GeoReferenceStation, IceCoordinateSystem, GeoPositionData
from datetime import datetime, timedelta
import os
from loguru import logger
from scipy.signal import medfilt, convolve,find_peaks
from pathlib import Path
import matplotlib.pylab as plt

class ALSPointCloudFilter(object):
    """ Base class for point cloud filters """

    def __init__(self, **kwargs):
        self.cfg = kwargs


class AtmosphericBackscatterFilter(ALSPointCloudFilter):
    """
    Identifies and removes target echoes that are presumably within the atmosphere
    based on elevation statistics for each line.
    """

    def __init__(self, filter_threshold_m=5):
        """
        Initialize the filter.
        :param filter_threshold_m:
        """
        super(AtmosphericBackscatterFilter, self).__init__(filter_threshold_m=filter_threshold_m)

    def apply(self, als):
        """
        Apply the filter for all lines in the ALS data container
        :param als:
        :return:
        """

        # Filter points outside the [-threshold, threshold] interval around the 
        # first mode of elevations
        elevations = als.get('elevation')
        # Determine elevation of first mode
        hist,bins = np.histogram(elevations[np.isfinite(elevations)],bins=100)
        diff = np.diff(np.append(np.zeros((1,)),hist))
        ind_peak = np.where(np.all([diff[1:]<0,diff[:-1]>0],axis=0))[0][0]
        min_mode_elev = np.mean(bins[ind_peak:ind_peak+2])
        threshold = 20
        # Mask points outside the interval
        mask = np.where(np.any([elevations>min_mode_elev+threshold,
                                elevations<min_mode_elev-threshold],axis=0))
        elevations[mask] = np.nan
        als.set("elevation", elevations)
        
        for line_index in np.arange(als.n_lines):

            # 1  Compute the median elevation of a line
            elevation = als.get("elevation")
            elevations = elevation[line_index, :]
            line_median = np.nanmedian(elevations)

            # 2. Fill nan values with median elevation
            # This is needed for spike detection
            elevations_nonan = np.copy(elevations)
            elevations_nonan[np.isnan(elevations_nonan)] = line_median

            # Search for sudden changes (spikes)
            spike_indices = self._get_filter_indices(elevations_nonan, self.cfg["filter_threshold_m"])

            # Remove spiky elevations
            elevation[line_index, spike_indices] = np.nan
            als.set("elevation", elevation)

    @staticmethod
    def _get_filter_indices(vector, filter_treshold):
        """ Compute the indices of potential spikes and save them in self.spike_indices """

        # Compute index-wise change in data
        diff = vector[1:] - vector[0:-1]

        # Compute change of data point to both directions
        diff_right = np.full(vector.shape, np.nan)
        diff_left = np.full(vector.shape, np.nan)
        diff_right[1:] = diff
        diff_left[0:-1] = diff

        # Check for data change exceeds the filter threshold
        right_threshold = np.abs(diff_right) > filter_treshold
        left_threshold = np.abs(diff_left) > filter_treshold

        # Check where data point is local extrema
        is_local_extrema = np.not_equal(diff_right > 0, diff_left > 0)
        condition1 = np.logical_and(right_threshold, left_threshold)

        # point is spike: if change on both sides exceeds threshold and is local
        # extrema
        is_spike = np.logical_and(condition1, is_local_extrema)
        spike_indices = np.where(is_spike)[0]

        return spike_indices


class IceDriftCorrection(ALSPointCloudFilter):
    """
    Corrects for ice drift during data aquisition, using floenavi or Polarstern position
    """

    def __init__(self,use_polarstern=False):
        """
        Initialize the filter.
        :param filter_threshold_m:
        """
        super(IceDriftCorrection, self).__init__(use_polarstern=use_polarstern)

    def apply(self, als):
        """
        Apply the filter for all lines in the ALS data container
        :param als:
        :return:
        """

        logger.info("IceDriftCorrection is applied")
        # 1. Initialise IceDriftStation
        self._get_IceDriftStation(als,use_polarstern=self.cfg["use_polarstern"])

        # 2. Initialise empty x,y arrays in als for the projection
        als.init_IceDriftCorrection()

        # 3. mask nan values for faster computation
        nonan = np.where(np.logical_or(np.isfinite(als.get("longitude")), np.isfinite(als.get("latitude"))))

        # 4. Generate GeoPositionData object from als
        time_als = np.array([datetime(1970,1,1,0,0,0) + timedelta(0,isec) for isec in als.get("timestamp")[nonan]])
        als_geo_pos = GeoPositionData(time_als,als.get("longitude")[nonan],als.get("latitude")[nonan])

        # 5. Compute projection
        icepos = self.IceCoordinateSystem.get_xy_coordinates(als_geo_pos)

        # 6. Store projected coordinates
        als.x[nonan] = icepos.xc
        als.y[nonan] = icepos.yc

        # 7. Set IceDriftCorrected
        als.IceDriftCorrected = True
        als.IceCoordinateSystem = self.IceCoordinateSystem


    def _get_IceDriftStation(self,als,use_polarstern=False):
        # Check for master solutions of Leg 1-3 in floenavi package
        path_data = os.path.join('/'.join(floenavi.__file__.split('/')[:-2]),'data/master-solution')
        ms_sol = np.array([ifile for ifile in os.listdir(path_data) if ifile.endswith('.csv')])
        ms_sol_dates = np.array([[datetime.strptime(ifile.split('-')[2],'%Y%m%d'),
                                  datetime.strptime(ifile.split('-')[3],'%Y%m%d')] for ifile in ms_sol])
        ind_begin = np.where(np.logical_and(als.tcs_segment_datetime>=ms_sol_dates[:,0],
                                            als.tcs_segment_datetime<=ms_sol_dates[:,1]))[0]
        ind_end   = np.where(np.logical_and(als.tce_segment_datetime>=ms_sol_dates[:,0],
                                            als.tce_segment_datetime<=ms_sol_dates[:,1]))[0]
        self.read_floenavi = False
        if not use_polarstern:
            if ind_begin.size>0 and ind_end.size>0:
                if ind_begin==ind_end:
                    self.read_floenavi = True
                
        if self.read_floenavi:
            refstat_csv_file = os.path.join(path_data,ms_sol[ind_begin][0])
            refstat = GeoReferenceStation.from_csv(refstat_csv_file)
        else:
            refstat = PolarsternAWIDashboardPos(als.tcs_segment_datetime,als.tce_segment_datetime).reference_station
        
        self.IceCoordinateSystem = als.IceCoordinateSystem = IceCoordinateSystem(refstat)

        

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

    def apply(self, als,do_plot=False):
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
        self._export_open_water_points(peaks_glob,als)
        
  
    def _export_open_water_points(self, peaks_glob, als):
        self._get_export().write('test\n')
        
        
    def _get_export(self):
        # Check if file exists
        export_file = Path(self.cfg["export_file"]).absolute()
        if not export_file.is_file():
            # Otherwise create new file with header
            with export_file.open(mode='w') as f:
                f.write('header\n')
                f.close()
        # Open file to read
        return export_file.open(mode='a')
        
