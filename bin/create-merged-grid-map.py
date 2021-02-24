# -*- coding: utf-8 -*-

import os

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray
from loguru import logger
from matplotlib.colors import LightSource


def create_merged_grid_map():
    """
    Create a png from a merged ALS grid
    :return:
    """

    # directory = r"D:\ASIRAS\MOSAIC_2019\product-als\20191119_01_PS122-1_8-23_Heli-PS\level_l4_nc"
    # filename = r"20191119_als_merged_grid-stere.nc"

    # directory = r"D:\ASIRAS\MOSAIC_2019\product-als\20191206_01_PS122-1_10-78_Heli-PS\level_l4_nc"
    # filename = r"20191206_als_merged_grid-stere.nc"
    # output = "floe-alsgrid-20191206-map.png"

    directory = r"E:\ASIRAS\MOSAIC_2019\product-als\20200915_01_PS122-5_62-67_Heli-PS\level_l4_nc"
    filename = r"20200915_als_merged_grid-stere.nc"
    output = "floe-alsgrid-20200915-map.png"

    filepath = os.path.join(directory, filename)
    merged_l4 = ALSMergedGridNC(filepath)

    plot = ALSMergedGridMap(merged_l4)
    plot.savefig(output)


class ALSMergedGridNC(object):

    def __init__(self, filepath):
        """
        Create an ALS merged grid mockup instance from a netcdf
        :param filepath:
        """

        self.filepath = filepath
        self.nc = xarray.open_dataset(self.filepath)



    @property
    def xc(self):
        return self.nc.xc.values

    @property
    def yc(self):
        return self.nc.yc.values

    @property
    def xc_bounds(self):
        return [np.nanmin(self.xc), np.nanmax(self.xc)]

    @property
    def yc_bounds(self):
        return [np.nanmin(self.yc), np.nanmax(self.yc)]

    @property
    def width(self):
        bounds = self.xc_bounds
        return bounds[1]-bounds[0]

    @property
    def height(self):
        bounds = self.yc_bounds
        return bounds[1]-bounds[0]

    @property
    def grid(self):
        return self.nc.elevation.values

class ALSMergedGridMap(object):

    # --- Target Colors ---
    # AWI eisblau #00ace5
    # AWI tiefblau #003e6e
    # AWI grau 1 #4b4b4d
    # AWI grau 2 #bcbdbf

    def __init__(self, dem):
        """
        Object for creating plots (interactive, or png output) from gridded ALS DEM's
        """

        # --- Store Arguments ---

        # The DEM object
        self.dem = dem

        # Switch default font to Arial
        mpl.rcParams['font.sans-serif'] = "arial"

        # Basic setup of the figure
        self._init_figure()

    def savefig(self, filename, dpi=600):
        """
        Save the figure as png
        :param filename: (str) target filename (full filepath)
        :param dpi: (int) resolution (dots per inch)
        :return: None
        """
        self._create_figure()
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="black")
        plt.clf()

    def _init_figure(self):
        """ Basic figure properties (size and axes locations """

        self.fig = plt.figure("ALS DEM Map", figsize=(30, 40), facecolor='black')
        self.ax = plt.gca()
        self.ax.set_aspect("equal")

    def _create_figure(self):
        """
        Workflow of assembling the figure elements
        :return: None
        """

        # Add the main plot
        self._plot_dem()

    def _plot_dem(self):
        """
        Add the DEM plot and set its style
        :return:
        """

        # limits and colors
        xmin, xmax = np.array([0, self.dem.width])   # - ps_bridge_position[0]
        ymin, ymax = np.array([0, self.dem.height])  # - ps_bridge_position[1]
        data_extent = [xmin, xmax, ymin, ymax]
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)

        # --- Plot the DEM ---

        # Preset axis range
        self.ax.plot()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        logger.info("Shading DEM")
        rgba = self._get_image_object()
        logger.info("Plotting DEM")
        self.ax.imshow(rgba, interpolation='none', origin='lower', extent=data_extent, zorder=100)
        self.ax.set_facecolor("1.0")
        # label_color = "#4d4b4b"
        label_color = "0.9"
        label_color_2 = "#4d4b4b"

        # logger.info("Plot Ship")
        # ps_aft_position = (3640-ps_bridge_position[0], 2930-ps_bridge_position[1])
        # ps_length = 117.91
        # ship_heading = 229.
        # ps_bow_delta_x = ps_length * np.sin(np.deg2rad(ship_heading))
        # ps_bow_delta_y = ps_length * np.cos(np.deg2rad(ship_heading))
        # plt.arrow(ps_aft_position[0], ps_aft_position[1], ps_bow_delta_x, ps_bow_delta_y,
        #           length_includes_head=True,
        #           color=label_color, width=10, head_width=20, zorder=500)
        #
        # logger.info("Plot Bridge Position")
        # self.ax.scatter(ps_bridge_position[0], ps_bridge_position[1], s=40, marker="P", edgecolors=label_color,
        #                 color="none")

        # logger.info("Plot Range Rings")
        # angles = np.linspace(0, 2. * np.pi, 1000)
        # for i, rng in enumerate(np.arange(250, 4001, 250)):
        #
        #     if (i+1) % 4 == 0:
        #         line_props = dict(lw=0.75, color=label_color, alpha=0.75, zorder=300)
        #     else:
        #         line_props = dict(lw=0.75, linestyle="--", dashes=(5, 5), alpha=0.75, color=label_color, zorder=300)
        #
        #     x = rng * np.sin(angles)
        #     y = rng * np.cos(angles)
        #     self.ax.plot(x, y, **line_props)
        #
        #     if rng >= np.abs(xlim[0]):
        #         continue
        #
        #     label_angle = np.deg2rad(225)
        #     lx, ly = rng * np.sin(label_angle), rng*np.cos(label_angle)
        #     self.ax.annotate("%.0fm" % rng, (lx, ly), color=label_color, fontsize=8,
        #                      weight='bold', ha='center', va='center',
        #                      bbox=dict(boxstyle="round", fc="none", ec=label_color, lw=0.25),
        #                      zorder=600)

        # logger.info("Plot Clock Spines")
        # for i, clock_angle in enumerate(np.linspace(0, 330, 12)):
        #     x_distance = 4000. * np.sin(np.deg2rad(clock_angle + ship_heading))
        #     y_distance = 4000. * np.cos(np.deg2rad(clock_angle + ship_heading))
        #
        #     if i % 3 == 0:
        #         line_props = dict(lw=0.75, color=label_color, alpha=0.75, zorder=300)
        #     else:
        #         line_props = dict(lw=0.75, linestyle="--", dashes=(5, 5), alpha=0.75, color=label_color, zorder=300)
        #     self.ax.plot([0, x_distance], [0, y_distance],  **line_props)
        #
        #     xl = 1000. * np.sin(np.deg2rad(clock_angle + ship_heading))
        #     yl = 1000. * np.cos(np.deg2rad(clock_angle + ship_heading))
        #
        #     self.ax.annotate("%gh" % i, (xl, yl), color=label_color, fontsize=12,
        #                      weight='bold', ha='center', va='center',
        #                      bbox=dict(boxstyle="round",
        #                      fc="black", ec=label_color, lw=0.25),
        #                      zorder=600)
        #
        # spines = ["left", "top", "right", "bottom"]
        # for spine in spines:
        #     self.ax.spines[spine].set_color("1.0")

        grid_color = "#4b4d4d"
        grid_color = "0.9"
        for x_major_tick in np.arange(500, self.dem.width, 500):
            self.ax.plot([x_major_tick, x_major_tick], [0, self.dem.height],
                         color=grid_color, lw=0.25, zorder=200, alpha=0.5)
        for x_minor_tick in np.arange(100, self.dem.width, 100):
            self.ax.plot([x_minor_tick, x_minor_tick], [0, self.dem.height],
                         linestyle="--", dashes=(5, 5),
                         color=grid_color, lw=0.1, zorder=200, alpha=0.5)

        for y_major_tick in np.arange(500, self.dem.height, 500):
            self.ax.plot([0, self.dem.width], [y_major_tick, y_major_tick],
                         color=grid_color, lw=0.25, zorder=200, alpha=0.5)
        for y_minor_tick in np.arange(100, self.dem.height, 100):
            self.ax.plot([0, self.dem.width], [y_minor_tick, y_minor_tick],
                         linestyle="--", dashes=(5, 5),
                         color=grid_color, lw=0.1, zorder=200, alpha=0.5)

        # --- Axes Style ---
        major, minor = 500, 100
        self.ax.set_aspect('equal')
        for ax in [self.ax]:
            ax.tick_params(which='major', length=8, color=1.0, labelcolor=1.0)
            ax.tick_params(which='minor', length=4, color=1.0, labelcolor=1.0)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(major))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(major))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor))

            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")

            for target in ["xaxis", "yaxis"]:
                subax = getattr(ax, target)
                subax.label.set_color("1.0")
                subax.label.set_fontsize(14)

        self.ax.set_aspect("equal")
        self.ax.tick_params(axis='both', which="both",                  # Targets (both axis, minor+major)
                            color="1.0", width=1,                       # tick props
                            labelcolor="1.0", labelsize=14, pad=6)      # label props

    def _get_image_object(self):
        """
        Return a shaded image using the matplotlib lightsource based shading engine
        :return: rgb array
        """

        # Hill shade props
        lightsource_default_props = {'azdeg': 315.0, 'altdeg': 45.0}
        shade_default_props = {'blend_mode': 'soft', 'vert_exag': 2.5}

        # Colormap
        cmap = truncate_colormap(cmocean.cm.ice, 0.25, 0.95)
        cmap.set_bad("white")

        # nodata = np.where(np.isnan(self.dem.grid))
        # self.dem.grid[nodata] = 100

        vmin, vmax = [-0.25, 1.0]
        ls = LightSource(**lightsource_default_props)
        rgb = ls.shade(self.dem.grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       dx=0.5, dy=0.5, **shade_default_props)

        # nodata_indices = np.where(rgb == [np.nan, np.nan, np.nan, 1])
        # for nodata_index in nodata_indices:
        #     rgb[nodata_index] = [0.9, 0.9, 0.9, 1.0]

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


if __name__ == "__main__":
    create_merged_grid_map()
