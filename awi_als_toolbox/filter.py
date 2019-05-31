# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"


import numpy as np

class ALSPointCloudFilter(object):
    """ Base class for point cloud filters """

    def __init__(self, **kwargs):
        self.cfg = kwargs


class AtmosphericBackscatterFilter(ALSPointCloudFilter):
    """ A filter for removing backscatter from fog/ice crystals/ ... """

    def __init__(self, filter_threshold_m=2):
        """

        :param filter_threshold_m:
        """

        super(AtmosphericBackscatterFilter, self).__init__(filter_threshold_m=filter_threshold_m)

    def apply(self, als):
        """
        Line-wise outlier filter
        :param als:
        :return:
        """

        # import matplotlib.pyplot as plt
        # x = np.arange(als.n_shots)

        # The filter work linewise
        for line_index in np.arange(als.n_lines):

            # 1  Compute the median elevation of a line
            elevations = als.elevation[line_index, :]
            line_median = np.nanmedian(elevations)

            # plt.figure(dpi=150)
            # plt.scatter(x, als.elevation[line_index, :], s=1, edgecolors="none")

            # 2. Fill nan values with median elevation
            # This is needed for spike detection
            elevations_nonan = np.copy(elevations)
            elevations_nonan[np.isnan(elevations_nonan)] = line_median

            # Search for sudden changes (spikes)
            spike_indices = self._get_filter_indices(elevations_nonan, 2.0)

            # plt.scatter(x[spike_indices], als.elevation[line_index, spike_indices], s=2, edgecolor="red", c="none")
            # plt.plot(x, np.full(x.shape, line_median))
            # plt.show()

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
