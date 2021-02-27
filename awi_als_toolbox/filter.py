# -*- coding: utf-8 -*-

"""
A module containing filter algorithm for the ALS point cloud data.
"""

__author__ = "Stefan Hendricks"


import numpy as np


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

        for line_index in np.arange(als.n_lines):

            # 1  Compute the median elevation of a line
            elevations = als.elevation[line_index, :]
            line_median = np.nanmedian(elevations)

            # 2. Fill nan values with median elevation
            # This is needed for spike detection
            elevations_nonan = np.copy(elevations)
            elevations_nonan[np.isnan(elevations_nonan)] = line_median

            # Search for sudden changes (spikes)
            spike_indices = self._get_filter_indices(elevations_nonan, self.cfg["filter_threshold_m"])

            # Remove spiky elevations
            als.elevation[line_index, spike_indices] = np.nan

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
