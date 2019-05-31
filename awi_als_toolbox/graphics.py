# -*- coding: utf-8 -*-

""" sub-package for creating graphical representation of laserscanner data """

__author__ = "Stefan Hendricks"


import numpy as np

import cmocean

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import LightSource



class AlsDemMap(object):

    DEFAULT_LABELS = {'xaxis': 'meter', 'yaxis': 'meter',
                      'zaxis': 'Elevation (meter)'}
    HILLSHADE_DEFAULT = {'scale': 0.1, 'azdeg': 165.0, 'altdeg': 45.0}
    CMAP_DEFAULTS = {'name': 'premiver2', 'range': 'percintels',
                     'qmin': 1.0, 'qmax': 99.0, 'vmin': 0.0, 'vmax': 2.0,
                     'nice_numbers': True}

    def __init__(self):
        self._has_dem = False
        self._use_hillshade = True
        self._hillshade_args = self.HILLSHADE_DEFAULT
        self._actual_heading = True
        self._cmap = truncate_colormap(cmocean.cm.ice, 0.25, 0.95)
        self._cmap_settings = self.CMAP_DEFAULTS
        self._label = self.DEFAULT_LABELS
        self.grid_spacing = 100

        self.fig = plt.figure("ALS DEM Map", figsize=(20, 10), facecolor='white')
        self._fig_reference_aspect = 3

    def set_dem(self, dem):
        # TODO: Remove, use self.alsdem instead
        self.dem_x = dem.dem_x
        self.dem_y = dem.dem_y
        self.dem_z = dem.dem_z_masked
        self.dem_mask = dem.dem_mask

    def set_cmap(self, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._cmap_settings.update(kwargs)

    def set_hillshade(self, onoff, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._use_hillshade = onoff
        self._hillshade_args.update(kwargs)

    def set_label(self, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._label.update(kwargs)

    def set_grid_spacing(self, spacing):
        self.grid_spacing = spacing

    def quickview(self):
        """ Creates a quick and dirty DEM plot using matplotlib """

        import seaborn as sns
        sns.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 1.5})
        self._plot()
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError:
            figManager.window.state('zoomed')
        plt.show()
        plt.close(self.fig)

    def save_fig(self, filename, dpi=300):
        import seaborn as sns
        sns.set_context("talk", font_scale=1.2)
        self._plot()
        plt.savefig(filename, dpi=dpi)
        plt.clf()

    def _plot(self):

        # 2 axis: 1: DEM, 2: Colorbar
        ax1 = self.fig.add_axes([0.07, 0.30, 0.90, 0.65])
        ax2 = self.fig.add_axes([0.52, 0.15, 0.45, 0.03])
        # limits and colors
        vmin, vmax = self._get_range()
        xlim, ylim, data_extent = self._scale_axes()
        # Plot the DEM
        ax1.plot()
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        rgba = self._get_image_object()
        ax1.imshow(rgba, interpolation='none', origin='lower',  extent=data_extent, zorder=100)

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(self.grid_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(self.grid_spacing))

        ax1.set_aspect('equal')
        ax1.xaxis.label.set_color('0.5')
        ax1.set_xlabel(self._label["xaxis"])
        ax1.yaxis.label.set_color('0.5')
        ax1.set_ylabel(self._label["yaxis"])
        ax1.tick_params(axis='both', colors='0.5')

        # Plot the colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=self._cmap, norm=norm, orientation='horizontal')
        cb1.set_label(self._label["zaxis"])

    def _scale_axes(self):
        # Get Data extent
        left = 0
        right = np.amax(self.dem_x) - np.amin(self.dem_x)
        height = np.amax(self.dem_y) - np.amin(self.dem_y)
        bottom = -0.5 * height
        top = 0.5 * height
        # Get figure extent
        figure_aspect = (right-left)/(top-bottom)
        left_off, right_off, bottom_off, top_off = 0.0, 0.0, 0.0, 0.0
        if figure_aspect > self._fig_reference_aspect:
            ref_height = (right-left) / self._fig_reference_aspect
            ref_height -= (top-bottom)
            bottom_off = -0.5*ref_height
            top_off = 0.5*ref_height
        if figure_aspect < self._fig_reference_aspect:
            ref_width = (top-bottom) * self._fig_reference_aspect
            ref_width -= (right-left)
            left_off = -0.5*ref_width
            right_off = 0.5*ref_width
        xlim = [left+left_off, right+right_off]
        ylim = [bottom+bottom_off, top+top_off]
        data_extent = [left, right, bottom, top]
        return xlim, ylim, data_extent

    def _get_percintels(self):
        """ Calculates the percintels of the elevation data """
        from plib.helpers.scaling import auto_bins
        finite = np.where(np.isfinite(self.dem_z))
        qmin = self._cmap_settings['qmin']
        qmax = self._cmap_settings['qmax']
        limits = np.percentile(self.dem_z[finite], [qmin, qmax])
        if self._cmap_settings['nice_numbers']:
            limits = auto_bins(limits[0], limits[1])
        return [np.amin(limits), np.amax(limits)]

    def _get_range(self):
        # TODO: Allow other modes (fixed range etc...)
        if self._cmap_settings['range'] == 'percintels':
            vmin, vmax = self._get_percintels()
        else:
            vmin = self._cmap_settings['vmin']
            vmax = self._cmap_settings['vmax']
        return vmin, vmax

    def _get_image_object(self):
        # TODO: Documentation
        vmin, vmax = self._get_range()
        # TODO: Allow hillshading configuration
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(self.dem_z, cmap=self._cmap, blend_mode="soft", vmin=vmin, vmax=vmax, vert_exag=10,
                       dx=0.25, dy=0.25)
        return rgb


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Return a subset of a colormap as new colormap. From:
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

    :param cmap:
    :param minval:
    :param maxval:
    :param n:
    :return:
    """
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap