# -*- coding: utf-8 -*-


import tqdm
import struct

from loguru import logger
from collections import OrderedDict
from cached_property import cached_property

import numpy as np
from datetime import datetime
from construct import Struct, Array, Double, Single, Byte


class AirborneLaserScannerFile(object):
    """ Class to retrieve data from a AWI ALS binary data file """

    def __init__(self, filepath, **header_kwargs):
        """
        Connects to a AWI binary ALS data file. The data is not parsed into memory when this is class is called,
        only the header information that is necessary to decode the binary data structure.

        Usage:
        ======

            alsfile = AirborneLaserScannerFile(filename)
            als = alsfile.get_data(start, stop)

        :param filepath: (str) The path of the AWI ALS file
        :param header_kwargs: Keywords for the header parsing class
        """

        # Store Parameter
        self.filepath = filepath

        # Decode and store header information
        self.header = ALSFileHeader(filepath, **header_kwargs)
        self.line_timestamp = None

        # Validate header infos
        if self.header.status != 0:
            msg = "Invalid header in %s\n" % filepath
            msg += "Red flags:\n"
            for errmsg in self.header.status_context.split(";")[:-1]:
                msg += " - %s\n" % errmsg
            header_dict = self.header.get_header_dict()
            msg += "Full Header:\n"
            for key in header_dict.keys():
                value = getattr(self.header, key)
                msg += "- %s:%s\n" % (key, str(value))
            raise IOError(msg)

        # Read the line timestamp
        # -> on timestamp per line to later select subsets of the full content
        self._read_line_timestamp()

        # Establish the line parser from the header information
        line_parser = self._get_line_parser(self.header.data_points_per_line)
        self.line_parser = line_parser.compile()

    def get_segment_list(self, segment_size_secs):
        """
        Get a list of (start, end) values for a given length (in seconds) that cover the entire profile
        :param segment_size_secs: (int) length of the segments in seconds
        :return: A list of (start, end) values in seconds
        """

        # Get the range of the file
        fstart = self.line_timestamp[0]
        fstop = self.line_timestamp[-1]

        # Get list of intervals
        ranges = np.arange(fstart, fstop+int(0.5*segment_size_secs), segment_size_secs)
        start_secs = ranges[:-1]
        end_secs = ranges[1:]

        if len(start_secs) == 0 and fstop > fstart:
            start_secs, end_secs = [fstart], [fstop]

        return list(zip(start_secs, end_secs))

    def get_data(self, start_seconds=None, end_seconds=None, sanitize=True):
        """
        Read a subset of the ALS data and return its content. The subset is selected with the (integer) seconds of
        the day. If `start_seconds` and `end_seconds` are omitted, the maximum range will be used
        :param start_seconds: (int) Start of the subset in seconds of the day
        :param end_seconds: (int) End of the subset in seconds of the day
        :param sanitize: (bool) Flag whether to filter illegal entries (out of bound lat/lons)
        :return: an ALSData object containing the data subset
        """

        # Check input
        if start_seconds is None:
            start_seconds = self.line_timestamp[0]

        if end_seconds is None:
            end_seconds = self.line_timestamp[-1]

        # Sanity check
        self._validate_time_range(start_seconds, end_seconds)

        # Get the number of lines
        line_range, n_selected_lines = self._get_line_range(start_seconds, end_seconds)

        # Get the section of the file to read
        startbyte, nbytes = self._get_data_bytes(line_range)

        # Get the shape of the output array
        nlines, nshots = n_selected_lines, self.header.data_points_per_line

        # Get a data container based on the variable definition based on the file version
        # NOTE: There two types of variables per line.
        #       1) Variables for each echo (shot_vars with dimensions [n_lines, n_shots])
        #       2) Variables only stored once per line (line_vars with dimensions [n_lines])
        shot_vars, line_vars = self._get_data_container(nlines, nshots)

        # Read the binary data.
        # NOTE: This is done per line and not in bulk to allow the variable definition
        #       to be incomplete as the data content of earlier files is not always
        #       well defined. As a minimum the variables time, longitude, latitude and
        #       elevation should always be present and will be read correctly if the
        #       reader is always positioned at the start byte of a particular line
        with open(self.filepath, 'rb') as f:

            for i in tqdm.tqdm(np.arange(n_selected_lines), desc="Parse lines"):

                # Position to the start byte of the current line
                f.seek(startbyte)
                bindat = f.read(nbytes)

                # Parse the line bytes
                line_data = self.line_parser.parse(bindat)

                # Transfer parsed variables to data container
                for shot_var_name in shot_vars.keys():
                    shot_vars[shot_var_name][i, :] = line_data[shot_var_name]
                for line_var_name in line_vars.keys():
                    line_vars[line_var_name][i] = line_data[line_var_name]

                # Go to next line
                startbyte = np.uint64(startbyte + nbytes)

        # Convert timestamp (seconds since start of the UTC day -> seconds since 1970-01-01)
        shot_vars["timestamp"] = self.timestamp2time(shot_vars["timestamp"])

        # --- Create output object ---
        # Save the search time (both in original units and in seconds since epoch
        seconds = self.timestamp2time(np.array([start_seconds, end_seconds]))
        segment_window = [[seconds[0], start_seconds], [seconds[1], end_seconds]]

        # Init the data container and store debug data
        als = ALSPointCloudData(shot_vars, line_vars, segment_window=segment_window)
        als.set_debug_data(startbyte=startbyte, nbytes=nbytes, line_range=line_range)
        if sanitize:
            als.sanitize()

        # Done and return the data
        return als

    def timestamp2time(self, timestamp):
        """
        Convert the timestamp used in the ALS laserscanner files (seconds since start of the day) to
        a more standardized one (e.g., seconds since 1970-01-01).
        :param timestamp: time data from AWI ALS file
        :return: time: timestamp since target epoch
        """

        # Init the output array
        epoch_offset_seconds = (self.source_epoch - self.target_epoch).total_seconds()
        time = timestamp + epoch_offset_seconds

        return time

    def _validate_time_range(self, start, stop):
        """ Check for oddities in the time range selection """
        fstart = self.line_timestamp[0]
        fstop = self.line_timestamp[-1]

        # Raise Errors
        if start > stop:
            msg = "start time {start} after stop time {stop}".format(start=start, stop=stop)
            raise ValueError(msg)
        if start > fstop or stop < fstart:
            msg = "time range {start} - {stop} out of bounds {fstart} - {fstop}"
            msg = msg.format(start=start, stop=stop, fstart=fstart, fstop=fstop)
            raise ValueError(msg)

        # Raise Warnings
        if start < fstart:
            msg = "start time {start} before actual start of file {fstart}".format(start=start, fstart=fstart)
            logger.warning(msg)
        if stop > fstop:
            logger.warning("stop time {stop} after actual end of file {fstop}".format(stop=stop, fstop=fstop))

    def _get_data_container(self, n_lines, n_shots):
        """
        Return a data structure fitting the file data content
        :param n_lines:
        :param n_shots:
        :return:
        """
        # Start with an empty dictionary
        shot_vars = dict()

        # Add all per shot variables
        per_shot_variables = self.per_shot_variables
        for variable_name in per_shot_variables:
            dtypes = per_shot_variables[variable_name]
            shot_vars[variable_name] = np.ndarray((n_lines, n_shots), dtype=dtypes[0])

        # Add all per line variables
        # NOTE: This is only relevant for file version 2 and will not add any
        #       variables for file version 1
        line_vars = dict()
        per_line_variables = self.per_line_variables
        for variable_name in per_line_variables:
            dtypes = per_line_variables[variable_name]
            line_vars[variable_name] = np.ndarray(n_lines, dtype=dtypes[0])

        return shot_vars, line_vars

    def _get_line_parser(self, n_shots):
        """
        Create a construct structure parser based on the number of variables in the data file
        and the number of shots per line
        :param n_shots:
        :return:
        """

        # Start with an empty structure
        parser = Struct()

        # Add all per shot variables
        per_shot_variables = self.per_shot_variables
        for variable_name in per_shot_variables:
            dtypes = per_shot_variables[variable_name]
            parser = parser + Struct(variable_name / Array(n_shots, dtypes[1]))

        # Add all per line variables
        # NOTE: This is only relevant for file version 2 and will not add any
        #       variables for file version 1
        per_line_variables = self.per_line_variables
        for variable_name in per_line_variables:
            dtypes = per_line_variables[variable_name]
            parser = parser + Struct(variable_name / dtypes[1])

        return parser

    def _get_data_bytes(self, line_range):
        """
        Computes the start byte and the number of bytes to read for the given lines
        :param line_range: (array) index of first and last line to read
        :return: (int, int) startbyte and the number of bytes for the data subset
        """

        # Start byte of scan line
        startbyte = np.uint64(self.header.byte_size)
        startbyte += np.uint64(self.header.bytes_sec_line)
        startbyte += np.uint64(line_range[0]) * np.uint64(self.header.bytes_per_line)

        # Number bytes for selected scan lines
        nbytes = self.header.bytes_per_line

        return startbyte, nbytes

    def _get_line_range(self, start_seconds, end_seconds):
        """
        Identify the last and first line for the time range
        :param start_seconds: (int) start of the subset in seconds of the day
        :param end_seconds: (int) end of the subset in seconds of the day
        :return: (int list, int) a list with [first line index, last line index] and number of lines
        """

        # Get the number of lines
        line_range = [
            np.where(self.line_timestamp >= start_seconds)[0][0],
            np.where(self.line_timestamp < end_seconds)[0][-1]]
        n_selected_lines = line_range[1] - line_range[0] + 1

        logger.debug("Line Range: %d - %d" % (line_range[0], line_range[1]))

        return line_range, n_selected_lines

    def _read_line_timestamp(self):
        """ Read the line time stamp """
        with open(self.filepath, 'rb') as f:
            f.seek(self.header.byte_size)
            data = f.read(self.header.bytes_sec_line)
        struct_def = ">{scan_lines}L".format(scan_lines=self.header.scan_lines)
        self.line_timestamp = np.array(struct.unpack(struct_def, data))

    @property
    def source_epoch(self):
        """
        Return the epoch for the time definition in the ALS files (start of UTC day)
        :return: datetime
        """
        return datetime(int(self.header.year), int(self.header.month), int(self.header.day))

    @cached_property
    def per_shot_variables(self):
        """
        Return a list of variables for each shot per line and their data type
        The data type is defined as the tupe (numpy data type, construct data type)
        :return: OrderedDict
        """
        per_shot_variables = OrderedDict((('timestamp', (np.float64, Double)),
                                          ('longitude', (np.float64, Double)),
                                          ('latitude', (np.float64, Double)),
                                          ('elevation', (np.float64, Double))))
        return per_shot_variables

    @cached_property
    def per_line_variables(self):
        """
        Return a list of variables for each line and their data type
        The data type is defined as the tupe (numpy data type, construct data type)
        :return: OrderedDict
        """
        return OrderedDict()

    @cached_property
    def target_epoch(self):
        """
        The
        :return: datetime.datetime
        """
        return datetime(1970, 1, 1)


class AirborneLaserScannerFileV2(AirborneLaserScannerFile):
    """
    Class to retrieve data from a AWI ALS binary data file in file version 2 (.alsbin2).
    This is a subclass of AirborneLaserScannerFile using the same mechanics but overwrites
    the variable definition
    """

    def __init__(self, *args, **kwargs):
        """
        Connects to a AWI binary ALS data file. The data is not parsed into memory when this is class is called,
        only the header information that is necessary to decode the binary data structure.

        Usage:
        ======

            alsfile = AirborneLaserScannerFileV2(filename)
            als = alsfile.get_data(start, stop)

        :param filepath: (str) The path of the AWI ALS file
        :param header_kwargs: Keywords for the header parsing class
        """
        super(AirborneLaserScannerFileV2, self).__init__(*args, **kwargs)

    @cached_property
    def per_shot_variables(self):
        """
        Return a list of variables for each shot per line and their data type
        The data type is defined as the tupe (numpy data type, construct data type)
        :return: OrderedDict
        """
        per_shot_variables = OrderedDict((('timestamp', (np.float64, Double)),
                                          ('latitude', (np.float64, Double)),
                                          ('longitude', (np.float64, Double)),
                                          ('elevation', (np.float64, Double)),
                                          ('elevation_reference', (np.float64, Double)),
                                          ('amplitude', (np.float32, Single)),
                                          ('reflectance', (np.float32, Single)),
                                          ('echo_width', (np.float32, Single)),
                                          ('n_echoes', (np.byte, Byte))))
        return per_shot_variables

    @cached_property
    def per_line_variables(self):
        """
        Return a list of variables for each line and their data type
        The data type is defined as the tupe (numpy data type, construct data type)
        :return: OrderedDict
        """
        per_line_variables = OrderedDict((('aircraft_latitude', (np.float64, Double)),
                                          ('aircraft_longitude', (np.float64, Double)),
                                          ('aircraft_altitude', (np.float32, Single)),
                                          ('aircraft_pitch', (np.float32, Single)),
                                          ('aircraft_roll', (np.float32, Single)),
                                          ('aircraft_true_heading', (np.float32, Single)),
                                          ('fov_min', (np.float32, Single)),
                                          ('fov_max', (np.float32, Single)),
                                          ('range_min', (np.float32, Single)),
                                          ('range_max', (np.float32, Single))))
        return per_line_variables


class ALSFileHeader(object):
    """
    Class for parsing and storing header information of binary AWI ALS data files
    Note: The header strucutre is constant for all binary file formats
    """

    # Header information of the form (variable_name, [number of bytes, struct format])
    header_dict = OrderedDict((('scan_lines', [4, '>L']),
                               ('data_points_per_line', [2, '!H']),
                               ('bytes_per_line', [2, '>H']),
                               ('bytes_sec_line', [8, '>Q']),
                               ('year', [2, '>H']),
                               ('month', [1, '>b']),
                               ('day', [1, '>b']),
                               ('start_time_sec', [4, '>L']),
                               ('stop_time_sec', [4, '>L']),
                               ('device_name', [8, '>8s'])))

    def __init__(self, filepath, device_name_override=None):
        """
        Decode and store header information from binary AWI ALS files
        :param filepath: (str) The path to the ALS file
        :param device_name_override: (str, default: None) The name of the sensor. May be not correct in the source
        files for newer versions
        """

        self.device_name_override = device_name_override
        self._header_info_dict = {}
        self._parse_header(filepath)

        # This is legacy code
        if device_name_override is not None:
            self._header_info_dict["device_name"] = self.device_name_override

    def _parse_header(self, filepath):
        """
        Read the header
        :param filepath:
        :return:
        """
        # Read the header
        with open(filepath, 'rb') as f:

            # Read header size
            self.byte_size = struct.unpack('>b', f.read(1))[0]
            logger.info("als_header.byte_size: %s" % str(self.byte_size))
            if self.byte_size == 36:
                self.header_dict['data_points_per_line'] = [1, '>B']
            elif self.byte_size == 37:
                self.header_dict['data_points_per_line'] = [2, '>H']
            elif self.byte_size == 39:
                self.header_dict['bytes_per_line'] = [4, '>L']
            else:
                msg = "Unknown ALS L1B header size: %g (Should be 36, 37 or 39)"
                msg = msg % self.byte_size,
                raise ValueError(msg)

            # Read Rest of header
            for key in self.header_dict.keys():
                nbytes, fmt = self.header_dict[key][0], self.header_dict[key][1]
                value = struct.unpack(fmt, f.read(nbytes))[0]
                self._header_info_dict[key] = value

        # Check if the header was parsed correctly
        self._status = 0
        self._status_context = ""
        try:
            self._validate()
        except AttributeError:
            self._status = 1
            self._status_context += "Unhandled exception in validation;"

    @classmethod
    def get_header_dict(cls):
        return cls.header_dict

    def _validate(self):
        """
        Runs a series of plausibility tests (looking for a red flag) to check if the header information
        is legit. The reason is that the information will be garbage if one byte is off.
        This method will set the status flag to 1 if a red flag is found and add reasons to the status context
        :return: None
        """

        # 1. Test if start|stop_time_sec are within a day
        for targ in ["start_time_sec", "stop_time_sec"]:
            val = getattr(self, targ)
            if val < 0 or val > 86400:
                self._status = 1
                self._status_context += "%s out of bounds (%g);" % (targ, val)

    @property
    def center_beam_index(self):
        """ Returns index of the center beam """
        return int(np.median(np.arange(self.data_points_per_line)))

    @property
    def status(self):
        """ Status flag (0: ok, 1: invalid) """
        return int(self._status)

    @property
    def status_context(self):
        """ Status flag (0: ok, 1: invalid) """
        return self._status_context

    def __getattribute__(self, attr):
        """
        Modify the attribute getter to provide a shortcut to the
        content of the data frome
        :param attr:
        :return:
        """
        try:
            return super().__getattribute__(attr)
        except AttributeError as e:
            if attr in list(self._header_info_dict.keys()):
                return self._header_info_dict[attr]
            else:
                raise e


class ALSPointCloudData(object):
    """
    A data class container for ALS data extracted from binary point cloud data
    """

    def __init__(self, shot_vars, line_vars, segment_window=None):
        """
        Data container for ALS data ordered in scan lines.
        :param shot_vars:
        :param line_vars:
        :param segment_window:
        """

        # Add Metadata
        self.metadata = ALSMetadata()
        self.debug_data = {}
        self.segment_window = segment_window

        # save data arrays
        self._shot_vars = shot_vars
        self._line_vars = line_vars
        
        # add new weights field
        self.set_weights()

        # Update the metadata now with the data in place
        self._set_metadata()

    def init_IceDriftCorrection(self):
        self.x = np.empty(self.get("longitude").shape)*np.NaN
        self.y = np.empty(self.get("longitude").shape)*np.NaN
        self.IceDriftCorrected   = False
        self.IceCoordinateSystem = None

    def set_debug_data(self, **kwargs):
        self.debug_data.update(kwargs)

    def sanitize(self):
        """ Run a series of test to identify illegal data points (e.g. out of bounds lon/lat, etc) """

        # Find illegal latitude values
        latitude = self.get("latitude")
        longitude = self.get("longitude")
        illegal_lat = np.where(np.abs(latitude) > 90.0)
        illegal_lon = np.where(np.abs(longitude) > 180.0)
        if illegal_lat[0].size == 0 and illegal_lon[0].size == 0:
            return
        for illegal_values in [illegal_lat, illegal_lon]:
            for key in self.shot_variables:
                var = getattr(self, key)
                # FIXME: This will break for non float variable such as n_echoes
                var[illegal_values] = np.nan
                setattr(self, key, var)

    def _set_metadata(self):
        """
        Get metadata from the data arrays
        :return:
        """

        # Data type is fixed
        self.metadata.set_attribute("cdm_data_type", "point")

        # Compute geospatial parameters
        lat_min, lat_max = self.lat_range
        self.metadata.set_attribute("geospatial_lat_min", lat_min)
        self.metadata.set_attribute("geospatial_lat_max", lat_max)
        lon_min, lon_max = self.lon_range
        self.metadata.set_attribute("geospatial_lon_min", lon_min)
        self.metadata.set_attribute("geospatial_lon_max", lon_max)
        elev_min, elev_max = self.elev_range
        self.metadata.set_attribute("geospatial_vertical_min", elev_min)
        self.metadata.set_attribute("geospatial_vertical_max", elev_max)

        # Compute time parameters
        timestamp = self.get("timestamp")
        tcs = datetime.utcfromtimestamp(float(np.nanmin(timestamp)))
        tce = datetime.utcfromtimestamp(float(np.nanmax(timestamp)))
        self.metadata.set_attribute("time_coverage_start", tcs)
        self.metadata.set_attribute("time_coverage_end", tce)

    @property
    def dims(self):
        elevation = self.get("elevation")
        return elevation.shape

    @property
    def n_lines(self):
        return self.dims[0]

    @property
    def n_shots(self):
        return self.dims[1]

    @property
    def shot_variables(self):
        return self._shot_vars.keys()

    @property
    def line_variables(self):
        return self._line_vars.keys()

    @property
    def lat_range(self):
        latitude = self.get("latitude")
        return np.nanmin(latitude), np.nanmax(latitude)

    @property
    def lon_range(self):
        longitude = self.get("longitude")
        return np.nanmin(longitude), np.nanmax(longitude)

    @property
    def elev_range(self):
        elevation = self.get("elevation")
        return np.nanmin(elevation), np.nanmax(elevation)

    @property
    def has_valid_data(self):
        """ Returns a flag whether the object contains valid elevation data """
        elevation = self.get("elevation")
        return np.count_nonzero(np.isfinite(elevation)) > 0

    @property
    def segment_seconds(self):
        """
        Return the search window in seconds
        :return: Depends on whether keyword `segment_window` was set during the initialization
                 of the object. If yes, this property will contain the integer seconds since the
                 start of the day that have been used to extract the segment from the ALS file.
                 If not, a rounded version of the actual seconds of the segment will be returned
                 This will be incorrect if data is missing on either start or end of the segment.
        """
        return [self.tcs_segment_seconds, self.tce_segment_seconds]

    @property
    def segment_time(self):
        """
        Return the search window as datetime object
        :return: Depends on whether keyword `segment_window` was set during the initialization
                 of the object. If yes, this property will be derived from the integer seconds
                 that have been used to extract the segment from the ALS file. If not, a rounded
                 version of the actual seconds will be used to estimate the time that has been
                 used to extract the segment. This will be incorrect if data is missing on either
                 start or end of the segment.
        """
        return [datetime.utcfromtimestamp(self.tcs_segment_time),
                datetime.utcfromtimestamp(self.tce_segment_time)]

    @property
    def ref_time(self):
        tcs, tce = self.tcs_segment_time, self.tce_segment_time
        return tcs + 0.5*(tce-tcs)

    @property
    def time_bnds(self):
        tcs, tce = self.tcs_segment_time, self.tce_segment_time
        return [tcs, tce]

    @property
    def tcs_segment_time(self):
        """
        Return the segment start time in seconds since epoch
        :return: (double) seconds since epoch
        """
        timestamp = self.get("timestamp")
        if self.segment_window is None:
            return np.floor(np.nanmin(timestamp))
        else:
            return self.segment_window[0][0]

    @property
    def tce_segment_time(self):
        """
        Return the segment end time in seconds since epoch
        :return: (double) seconds since epoch
        """
        timestamp = self.get("timestamp")
        if self.segment_window is None:
            return np.floor(np.nanmax(timestamp))
        else:
            return self.segment_window[1][0]

    @property
    def tcs_segment_datetime(self):
        """
        Return the segment start time in seconds since epoch
        :return: (double) seconds since epoch
        """
        return datetime.utcfromtimestamp(self.tcs_segment_time)

    @property
    def tce_segment_datetime(self):
        """
        Return the segment end time in seconds since epoch
        :return: (double) seconds since epoch
        """
        return datetime.utcfromtimestamp(self.tce_segment_time)

    @property
    def tcs_segment_seconds(self):
        """
        Return the segment start time in seconds since start of day
        :return: (double) seconds from ALS file
        """
        timestamp = self.get("timestamp")
        if self.segment_window is None:
            return np.floor(np.nanmin(timestamp))
        else:
            return self.segment_window[0][1]

    @property
    def tce_segment_seconds(self):
        """
        Return the segment end time in seconds since start of day
        :return: (double) seconds from ALS file
        """
        timestamp = self.get("timestamp")
        if self.segment_window is None:
            return np.floor(np.nanmax(timestamp))
        else:
            return self.segment_window[1][1]

    @property
    def grid_variable_names(self):
        """
        Return a list of variables that can be gridded
        -> line variables except time, longitude, latitude
        :return:
        """
        grid_variables = list(self.shot_variables)
        for non_grid_variable in ["longitude", "latitude"]:#["timestamp", "longitude", "latitude"]:
            try:
                grid_variables.remove(non_grid_variable)
            except ValueError:
                pass
        return grid_variables

    def get(self, attr):
        """
        Modify the attribute getter to provide a shortcut to the data content
        :param attr:
        :return:
        """
        if attr=='weights' and not attr in self.line_variables:
            self.set_weights()
        if attr in self.shot_variables:
            return self._shot_vars[attr]
        elif attr in self.line_variables:
            return self._line_vars[attr]
        else:
            return None

    def set(self, attr, var):
        """
        Modify the attribute getter to provide a shortcut to the data content
        :param attr:
        :param var:
        :return:
        """
        if attr in self.shot_variables:
            self._shot_vars[attr] = var
        elif attr in self.line_variables:
            self._line_vars[attr] = var
        else:
            return None
        
    def set_weights(self):
        """
        Set weights depending of angle of view
        """
        wght = ((1-np.linspace(0,1,self.dims[1]))*np.linspace(0,1,self.dims[1]))
        wght /= np.max(wght)
        wght = np.tile(wght,(self.dims[0],1))
        self._shot_vars['weights'] = wght


class ALSMetadata(object):
    """
    A container for product metadata following CF/Attribute Convention for Data Discovery 1.3
    -> http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery_1-3
    """

    ATTR_DICT = ["title", "summary", "keywords", "Conventions", "id", "naming_authority",
                 "history", "source", "processing_level", "comment", "acknowledgement", "license",
                 "standard_name_vocabulary", "date_created", "creator_name", "creator_url", "creator_email",
                 "institution", "project", "publisher_name", "publisher_url", "publisher_email",
                 "geospatial_bound", "geospatial_bounds_crs", "geospatial_bounds_vertical_crs",
                 "geospatial_lat_min",  "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max",
                 "geospatial_vertical_min", "geospatial_vertical_max", "time_coverage_start", "time_coverage_end",
                 "time_coverage_duration", "time_coverage_resolution", "creator_type", "creator_institution",
                 "publisher_type", "publisher_institution", "program", "contributor_name", "contributor_role",
                 "geospatial_lat_units", "geospatial_lat_resolution", "geospatial_lon_units",
                 "geospatial_lon_resolution", "geospatial_vertical_units", "geospatial_vertical_resolution",
                 "date_issued", "date_metadata_modified", "product_version", "platform", "platform_vocabulary",
                 "instrument", "instrument_vocabulary", "cdm_data_type", "metadata_link", "references"]

    def __init__(self):
        """ Product Metadata following Attribute Convention for Data Discovery 1.3 """

        # Init all attributes with None
        for key in self.ATTR_DICT:
            setattr(self, key, None)

        # Init variable attr dict
        # (contains variable attributes as a dictionary of variable name)
        self.var_attrs = {}

    def get_var_attrs(self, variable_name):
        """
        Retrieves a dictionary of variable attributes
        :param variable_name: The name of the variable
        :return: dictionary (empty of no definition found)
        """
        return self.var_attrs.get(variable_name, {})

    def set_attributes(self, global_attrs, **kwargs):
        """
        Batch set metadata attributes
        :param global_attrs: (dict) a dict of metadata attributes
        """

        for key in global_attrs.keys():
            self.set_attribute(key, global_attrs[key], **kwargs)

    def set_variable_attributes(self, variable_attrs):
        """ Set the dictionary for variable attributes """
        self.var_attrs.update(variable_attrs)

    def set_attribute(self, key, value, raise_on_error=True, datetime2iso8601=True):
        """
        Set an attribute of the metadata container.
        :param key: (str) the name of the attribute (must be in ATTR_DICT)
        :param value: the value of the attribute
        :param raise_on_error: (bool) flag whether a ValueError should be raised when key is not a valid attribute
                                      name
        :param datetime2iso8601: (bool)
        :return: None
        """

        if key in self.ATTR_DICT:
            if isinstance(value, datetime) and datetime2iso8601:
                value = value.isoformat()
            setattr(self, key, value)
        else:
            if raise_on_error:
                raise ValueError("invalid metadata attribute name: %s" % str(key))
            else:
                pass

    def copy(self):
        """ Returns a copy of the current metadata, e.g. if a derived product inherits part of the metadata """
        cls = ALSMetadata()
        cls.set_attributes(self.attribute_dict)
        cls.set_variable_attributes(self.var_attrs)
        return cls

    @property
    def attribute_dict(self):
        """ Returns an attribute dict (not None attributes only) """
        return {key: getattr(self, key) for key in self.ATTR_DICT if getattr(self, key) is not None}

    @property
    def items(self):
        return self.attribute_dict.items()
