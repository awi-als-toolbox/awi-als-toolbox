# -*- coding: utf-8 -*-

""" sub-package for creating graphical representation of laserscanner data """

__author__ = "Stefan Hendricks"

import numpy as np

import cmocean

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import LightSource

from mpl_toolkits.basemap import Basemap


class AlsDemMap(object):

    # --- Target Colors ---
    # AWI eisblau #00ace5
    # AWI tiefblau #003e6e
    # AWI grau 1 #4b4b4d
    # AWI grau 2 #bcbdbf

    def __init__(self, dem, cfg=None, logo_path=None):
        """
        Object for creating plots (interactive, or png output) from gridded ALS DEM's
        :param dem:
        :param cfg:
        """

        # --- Store Arguments ---

        # The DEM object
        self.dem = dem

        # Plot configuration (fallback to default if no cfg has been passed)
        if cfg is None:
            self.cfg = AlsDemMapCfg()
        else:
            self.cfg = cfg

        # Logo path (experimental)
        self.logo_path = logo_path
        self.map_extent = None

        # Switch default font to Arial
        mpl.rcParams['font.sans-serif'] = "arial"

        # Basic setup of the figure
        self._init_figure()


    def show(self):
        """
        Creates an interactive view of the figure (-> plt.show())
        :return:  None
        """

        self._create_figure()
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError:
            figManager.window.state('zoomed')
        plt.show()
        plt.close(self.fig)

    def savefig(self, filename, dpi=300):
        """
        Save the figure as png
        :param filename: (str) target filename (full filepath)
        :param dpi: (int) resolution (dots per inch)
        :return: None
        """
        self._create_figure()
        plt.savefig(filename, dpi=dpi)
        plt.clf()

    def _init_figure(self):
        """ Basic figure properties (size and axes locations """

        self.fig = plt.figure("ALS DEM Map", figsize=(20, 12), facecolor='white')
        self._fig_reference_aspect = 3

        # four axis: 1: DEM, 2: Colorbar
        self.ax_dem = self.fig.add_axes([0.05, 0.38, 0.90, 0.65])
        self.ax_cmap = self.fig.add_axes([0.6, 0.36, 0.35, 0.03])
        self.ax_globe = self.fig.add_axes([0.05, 0.07, 0.2, 0.32])
        self.ax_map = self.fig.add_axes([0.30, 0.07, 0.2, 0.32])

    def _create_figure(self):
        """
        Workflow of assembling the figure elements
        :return: None
        """

        # Add the main plot
        self._plot_dem()

        # Add the colorbar
        self._plot_cb()

        # Add the globe view showing position in large context
        self._plot_globe()

        # Add written metadata
        self._plot_metadata()

    def _plot_dem(self):
        """
        Add the DEM plot and set its style
        :return:
        """

        # limits and colors
        xlim, ylim, data_extent = self._scale_axes()

        # --- Plot the DEM ---

        # Preset axis range
        self.ax_dem.plot()
        self.ax_dem.set_xlim(xlim)
        self.ax_dem.set_ylim(ylim)

        rgba = self._get_image_object()
        self.ax_dem.imshow(rgba, interpolation='none', origin='lower',  extent=data_extent, zorder=100)
        self.ax_dem.set_facecolor("0.0")

        # --- Axes Style ---
        major, minor = self._get_tick_spacing()
        self.ax_dem.tick_params(which='major', length=8)
        self.ax_dem.tick_params(which='minor', length=4)
        self.ax_dem.xaxis.set_ticks_position('both')
        self.ax_dem.yaxis.set_ticks_position('both')
        self.ax_dem.xaxis.set_major_locator(ticker.MultipleLocator(major))
        self.ax_dem.yaxis.set_major_locator(ticker.MultipleLocator(major))
        self.ax_dem.xaxis.set_minor_locator(ticker.MultipleLocator(minor))
        self.ax_dem.yaxis.set_minor_locator(ticker.MultipleLocator(minor))

        self.ax_dem.set_aspect('equal')

        self.ax_dem.set_xlabel(self.cfg.get_label("xaxis"))
        self.ax_dem.set_ylabel(self.cfg.get_label("yaxis"))
        for target in ["xaxis", "yaxis"]:
            ax = getattr(self.ax_dem, target)
            ax.label.set_color("#4b4b4d")
            ax.label.set_fontsize(14)

        self.ax_dem.tick_params(axis='both', which="both",                      # Targets (both axis, minor+major)
                                color="#bcbdbf", direction="in", width=1,       # tick props
                                labelcolor="#4b4b4d", labelsize=14, pad=6)      # label props

    def _plot_cb(self):
        """
        Add the colobar object
        :return:
        """

        vmin, vmax = self.cfg.get_cmap_range(self.dem.dem_z_masked)

        # Plot the colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(self.ax_cmap, cmap=self.cfg.cmap, norm=norm, orientation='horizontal')
        cb1.set_label(self.cfg.get_label("zaxis"), fontsize=16, color="#4b4b4d")
        self.ax_cmap.tick_params(axis='both', color="#bcbdbf", labelcolor="#4b4b4d", labelsize=14)


    def _plot_globe(self):
        """
        Add an orthographic/full globe view with the marked position of the DEM segment
        :return:
        """

        # Get projection center from DEM
        lon_0, lat_0 = self.dem.get_swath_lonlat_center()
        m = Basemap(ax=self.ax_globe, projection='ortho', lon_0=lon_0, lat_0=lat_0, resolution='i')
        m.fillcontinents(color='#00ace5', lake_color='#00ace5')
        m.scatter(lon_0, lat_0, marker="x", color="#003e6e", latlon=True, zorder=100)
        m.drawmapboundary(color='#00ace5', linewidth=0.1)

    def _plot_metadata(self):
        """ Write metadata properties in the lower right corner of the plot"""

        batch_metadata = [("Project", "project"),
                          ("Platform", "platform"),
                          ("Sensor", "sensor")]

        metadata_props = dict(xycoords="figure fraction", color="#4b4b4d", fontsize=18, ha="left")
    def _get_tick_spacing(self):
        """
        Return the spacing for ticks (major and minor ticks) in axis units (m)
        :return: major, minor
        """
        max_side_len = self.dem.max_side_len

        major, minor = 50, 25
        if max_side_len > 500.:
            major, minor = 100, 50
        elif max_side_len > 1000.:
            major, minor = 200, 100

        return major, minor

    def _scale_axes(self):
        """
        Compute axes limits from axes aspect and data limits (Required for undistorted imshow)
        :return: values for xlim, ylim, data_extent
        """

        # Get Data extent
        left = 0
        right = np.amax(self.dem.dem_x) - np.amin(self.dem.dem_x)
        height = np.amax(self.dem.dem_y) - np.amin(self.dem.dem_y)
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
        elif figure_aspect <= self._fig_reference_aspect:
            ref_width = (top-bottom) * self._fig_reference_aspect
            ref_width -= (right-left)
            left_off = -0.5*ref_width
            right_off = 0.5*ref_width

        xlim = [left+left_off, right+right_off]
        ylim = [bottom+bottom_off, top+top_off]
        data_extent = [left, right, bottom, top]

        return xlim, ylim, data_extent

    def _get_image_object(self):
        """
        Return a shaded image using the matplotlib lightsource based shading engine
        :return: rgb array
        """
        vmin, vmax = self.cfg.get_cmap_range()
        ls = LightSource(**self.cfg.hillshape_props["ls_props"])
        rgb = ls.shade(self.dem.dem_z_masked, cmap=self.cfg.cmap, vmin=vmin, vmax=vmax,
                       dx=self.dem.cfg.resolution, dy=self.dem.cfg.resolution,
                       **self.cfg.hillshape_props["shade_props"])
        return rgb


class AlsDemMapCfg(object):
    """ Container for DEMMap plot configuration data """


    # --- Default Values ---

    # Axes Labels (Title has to be set separately)
    AX_DEFAULT_LABELS = {'xaxis': 'meter',
                         'yaxis': 'meter',
                         'zaxis': 'Elevation (meter)'}

    # Hillshade options (two properties: 1. Light source, 2. shading)
    LIGHTSOURCE_DEFAULT_PROPS = {'azdeg': 315.0, 'altdeg': 45.0}
    SHADE_DEFAULT_PROPS = {'blend_mode': 'soft', 'vert_exag': 10}

    # Color settings (color map, value range, etc)
    CMAP_DEFAULT_PROPS = {'name': 'default',       # currently only (default)
                          'range': 'percentile',   # [percentile|fixed]
                          # Properties for range -> percentile
                          'qmin': 1.0, 'qmax': 99.0, 'preset_range': None, 'preset_range_anchor': 'bottom',
                          'nice_numbers': True,
                          # Properties for range -> fixed
                          'vmin': 0.0, 'vmax': 1.0}

    def __init__(self, cmap_props=None, label_dict=None, hillshade_props=None):
        """
        Container for AlsDemMap plot configuration data
        :param cmap_props: (dict)
        :param label_dict: (dict)
        :param hillshade_props: (dict)
        """

        if cmap_props is None:
            self.cmap_props = self.CMAP_DEFAULT_PROPS
        else:
            self.cmap_props = cmap_props

        if label_dict is None:
            self.label_dict = self.AX_DEFAULT_LABELS
        else:
            self.label_dict = label_dict

        if hillshade_props is None:
            self.hillshape_props = dict(ls_props=self.LIGHTSOURCE_DEFAULT_PROPS,
                                        shade_props=self.SHADE_DEFAULT_PROPS)
        else:
            self.hillshape_props = hillshade_props

    @classmethod
    def preset(cls, name):

        if name == "elevation":
            return cls()
        elif name == "freeboard":

            # Use Freeboard as the colormap label
            label_dict = dict(cls.AX_DEFAULT_LABELS)
            label_dict["zaxis"] = "Freeboard (meter)"

            # For freeboard, the z range can be set as a fixed value
            # NOTE: the predefined range in the CMAP_DEFAULT_PROPS has already been selected
            #       for freeboard
            cmap_props = dict(cls.CMAP_DEFAULT_PROPS)
            cmap_props["range"] = "fixed"

            return cls(cmap_props=cmap_props, label_dict=label_dict)

    @property
    def cmap(self):
        """ Return the matplotlib cmap instance"""

        # The default colormap for all ALS DEM Plots
        # -> cropped cmocean sea ice colorbar
        if self.cmap_props["name"] == "default":
            cmap = truncate_colormap(cmocean.cm.ice, 0.25, 0.95)
        else:
            raise NotImplementedError("Unkown cmap name: %s" % self.cmap_props["name"])

        return cmap

    def get_cmap_range(self, *args):
        """
        Computes the range of the colormap (based on cmap_properties)
        :return:
        """

        # The default option for DEMS with a variable elevation range (e.g. l1b data)
        # -> vmin, vmax are estimated from the elevation distributions
        if self.cmap_props['range'] == 'percentile':
            vmin, vmax = self._get_percintels(*args)

        # Fixed elevation range, e.g. for freeboard
        elif self.cmap_props['range'] == 'fixed':
            vmin, vmax = self.cmap_props['vmin'], self.cmap_props['vmax']

        # Catch invalid requests
        else:
            raise NotImplementedError("Unkown cmap range mode: %s" % str(self.cmap_props['range']))

        return vmin, vmax


    def get_label(self, target):
        """
        Returns a label for a given axis
        :param target: (str) label target (xaxis, yaxis, zaxis)
        :return: (str) the label (will return empty string on invalid target
        """
        return self.label_dict.get(target, "")

    def _get_percintels(self, dem_z):
        """
        Calculates the percintels of the elevation data with the various options
        :param dem_z: Array with elevation/freeboard values
        :return: [vmin, vmax]: The computed value range of the colormap
        """

        # Compute percentiles from dem elevations
        finite = np.where(np.isfinite(dem_z))
        qmin, qmax = self.cmap_props['qmin'],  self.cmap_props['qmax']
        vmin, vmax = np.percentile(dem_z[finite], [qmin, qmax])

        # Modification option (Nice Numbers)
        if self.cmap_props['nice_numbers']:
            limits = auto_bins(vmin, vmax)
            vmin, vmax = np.amin(limits), np.amax(limits)

        # Modification options (Fixed range with variable min or max)
        if self.cmap_props['preset_range'] is not None:
            preset_range = self.cmap_props['preset_range']
            if self.cmap_props['preset_range_anchor'] == "bottom":
                vmax = vmin + preset_range
            elif self.cmap_props['preset_range_anchor'] == "top":
                vmin = vmax - preset_range
            else:
                msg = "invalid value for option `preset_range_anchor: %s [bottom|top]"
                msg = msg % str(self.cmap_props['preset_range_anchor'])
                return ValueError(msg)

        return vmin, vmax



def auto_bins(vmin, vmax, nbins=10):
    steps = [0.5, 1, 1.5, 2, 2.5, 4, 5, 6, 8, 10]
    scale, offset = scale_range(vmin, vmax, nbins)
    vmin -= offset
    vmax -= offset
    raw_step = (vmax-vmin)/nbins
    scaled_raw_step = raw_step/scale
    best_vmax = vmax
    best_vmin = vmin

    for step in steps:
        if step < scaled_raw_step:
            continue
        step *= scale
        best_vmin = step*divmod(vmin, step)[0]
        best_vmax = best_vmin + step*nbins
        if (best_vmax >= vmax):
            break
    return (np.arange(nbins+1) * step + best_vmin + offset)


def scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)
    maxabsv = max(abs(vmin), abs(vmax))
    if maxabsv == 0 or dv/maxabsv < 1e-12:
        return 1.0, 0.0
    meanv = 0.5*(vmax+vmin)
    if abs(meanv)/dv < threshold:
        offset = 0
    elif meanv > 0:
        ex = divmod(np.log10(meanv), 1)[0]
        offset = 10**ex
    else:
        ex = divmod(np.log10(-meanv), 1)[0]
        offset = -10**ex
    ex = divmod(np.log10(dv/n), 1)[0]
    scale = 10**ex
    return scale, offset


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