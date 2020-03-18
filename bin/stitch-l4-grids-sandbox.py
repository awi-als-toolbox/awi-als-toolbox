# -*- coding: utf-8 -*-

import os
import glob
import xarray

import numpy as np

from loguru import logger

from matplotlib.patches import Rectangle
import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import LightSource


def stitch_l4_grids_sandbox():

    # Lookup Directory for nc files
    mcs_copy_dir = r"D:\mcs\workspace\teams\aircraft_operations\heli-ps\riegl-vq580-s9999057"
    # flight_id = r"20191002_01_PS122-1_2-58_Heli-PS"
    flight_id = r"20191020_01_PS122-1_2-167_Heli-PS"
    l4_subdir = "level_l4_nc"

    # Get a list of l4 files
    l4_files_lookup = os.path.join(mcs_copy_dir, flight_id, l4_subdir, "awi-mosaic-l4-elevation*.nc")
    l4_files_all = sorted(glob.glob(l4_files_lookup))

    # Manually remove some files
    ignore_list = [0, 1, 4]
    l4_files = []
    for i, l4_file in enumerate(l4_files_all):
        if i in ignore_list:
            continue
        l4_files.append(l4_file)
    logger.info("Found %g l4 grid files" % len(l4_files))

    # Read the files
    l4_collect = ALSGridCollection(l4_files, res=0.5)
    merged_l4 = l4_collect.get_merged_grid()

    plot = ALSMergedGridMap(merged_l4)
    plot.savefig("floe-alsgrid-20191020-map.png")

    # plt.figure()
    # ax = plt.gca()
    # for i, grid in enumerate(l4_collect.grids):
    #     rect = Rectangle((grid.xc_bounds[0], grid.yc_bounds[0]),
    #                      grid.width, grid.height, ec="#76FF7A", fc="none", lw=1, alpha=0.5)
    #     plt.annotate(str(i), (grid.xcenter, grid.ycenter), va="center", ha="center")
    #     ax.add_patch(rect)
    #
    # plt.xlim(l4_collect.xc_bounds)
    # plt.ylim(l4_collect.yc_bounds)
    # ax.set_aspect('equal')
    # plt.show()


class ALSGridCollection(object):

    def __init__(self, filepaths, res=None, ignore_list=[]):
        self.filepaths = filepaths
        self.res = res
        self.ignore_list = ignore_list
        self.grids = []
        self._read_grid_data()

    def _read_grid_data(self):
        for filepath in self.filepaths:
            logger.info("Read: %s" % os.path.split(filepath)[-1])
            grid_data = ALSL4Grid(filepath)
            self.grids.append(grid_data)

    def get_merged_grid(self):
        x_min, x_max = self.xc_bounds
        y_min, y_max = self.yc_bounds
        merged_grid = ALSMergedGrid(x_min, x_max, y_min, y_max, self.res)
        logger.info("Merge Grids:")
        for i, grid in enumerate(self.grids):
            if i in self.ignore_list:
                continue
            if (i+1) % 10 == 0:
                logger.info("... %g / %g done" % (i+1, self.n_grids))
            merged_grid.add_grid(grid)
        logger.info("... %g / %g done" % (self.n_grids, self.n_grids))
        return merged_grid

    @property
    def xc_bounds(self):
        grid_min_bounds = [grid.xc_bounds[0] for grid in self.grids]
        grid_max_bounds = [grid.xc_bounds[1] for grid in self.grids]
        return [np.nanmin(grid_min_bounds), np.nanmax(grid_max_bounds)]

    @property
    def yc_bounds(self):
        grid_min_bounds = [grid.yc_bounds[0] for grid in self.grids]
        grid_max_bounds = [grid.yc_bounds[1] for grid in self.grids]
        return [np.nanmin(grid_min_bounds), np.nanmax(grid_max_bounds)]

    @property
    def n_grids(self):
        return len(self.grids)


class ALSL4Grid(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self._read_file()

    def _read_file(self):
        self.nc = xarray.open_dataset(self.filepath)

    @property
    def grid_xc_yc(self):
        xc, yc = self.nc.xc.values, self.nc.yc.values
        return np.meshgrid(xc, yc)

    @property
    def xc_bounds(self):
        xc = self.nc.xc.values
        return [np.nanmin(xc), np.nanmax(xc)]

    @property
    def yc_bounds(self):
        yc = self.nc.yc.values
        return [np.nanmin(yc), np.nanmax(yc)]

    @property
    def xcenter(self):
        return np.nanmean(self.xc_bounds)

    @property
    def ycenter(self):
        return np.nanmean(self.yc_bounds)

    @property
    def proj_extent(self):
        xc_bounds = self.xc_bounds
        yc_bounds = self.yc_bounds
        return [xc_bounds[0], yc_bounds[0], xc_bounds[1], yc_bounds[1]]

    @property
    def width(self):
        bounds = self.xc_bounds
        return bounds[1]-bounds[0]

    @property
    def height(self):
        bounds = self.yc_bounds
        return bounds[1]-bounds[0]

    @property
    def value(self):
        return self.nc.elevation.values


class ALSMergedGrid(object):

    def __init__(self, x_min, x_max, y_min, y_max, res_m):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res = res_m

        # Compute the shape of the full grid
        self.xc = np.linspace(self.x_min, self.x_max, (self.x_max-self.x_min) / res_m)
        self.yc = np.linspace(self.y_min, self.y_max, (self.y_max - self.y_min) / res_m)
        self.xy = np.meshgrid(self.xc, self.yc)
        self.dims = self.xy[0].shape
        self.grid = np.full(self.dims, np.nan)

    def add_grid(self, grid):

        # Compute the offset indices between merged grid and grid subset
        xi_offset = int((grid.xc_bounds[0]-self.xc_bounds[0])/self.res)
        yj_offset = int((grid.yc_bounds[0]-self.yc_bounds[0])/self.res)

        # Find finite values in the grid subset
        subset_valid_indices = np.where(np.isfinite(grid.value))

        subset_yj, subset_xi = subset_valid_indices[0].copy(), subset_valid_indices[1].copy()
        subset_yj += yj_offset
        subset_xi += xi_offset
        merged_valid_indices = (subset_yj, subset_xi)

        self.grid[merged_valid_indices] = grid.value[subset_valid_indices]-np.nanmean(grid.value)

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

    def savefig(self, filename, dpi=300):
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

        ps_bridge_position = (3560, 2860)

        # limits and colors
        xmin, xmax = np.array([0, self.dem.width]) # - ps_bridge_position[0]
        ymin, ymax = np.array([0, self.dem.height]) # - ps_bridge_position[1]
        data_extent = [xmin, xmax, ymin, ymax]
        # xlim = (-2500, 1500)
        # ylim = (-2000, 2000)
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

        for x_major_tick in np.arange(500, self.dem.width, 500):
            self.ax.plot([x_major_tick, x_major_tick], [0, self.dem.height],
                         color="#4b4d4d", lw=0.5, zorder=200, alpha=0.75)
        for x_minor_tick in np.arange(100, self.dem.width, 100):
            self.ax.plot([x_minor_tick, x_minor_tick], [0, self.dem.height],
                         linestyle="--", dashes=(5, 5),
                         color="#4b4d4d", lw=0.25, zorder=200, alpha=0.75)

        for y_major_tick in np.arange(500, self.dem.height, 500):
            self.ax.plot([0, self.dem.width], [y_major_tick, y_major_tick],
                         color="#4b4d4d", lw=0.5, zorder=200, alpha=0.75)
        for y_minor_tick in np.arange(100, self.dem.height, 100):
            self.ax.plot([0, self.dem.width], [y_minor_tick, y_minor_tick],
                         linestyle="--", dashes=(5, 5),
                         color="#4b4d4d", lw=0.25, zorder=200, alpha=0.75)

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
        shade_default_props = {'blend_mode': 'soft', 'vert_exag': 10}

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
    stitch_l4_grids_sandbox()
