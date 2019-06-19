# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"

import numpy as np

from collections import OrderedDict

import struct
import logging


class AirborneLaserScannerFile(object):
    """ Class to retrieve data from a AWI ALS binary data file """

    # Variable names and their data type
    line_variables = OrderedDict((('timestamp', np.float64),
                                  ('longitude', np.float64),
                                  ('latitude', np.float64),
                                  ('elevation', np.float64),
                                  ('amplitude', np.float32),
                                  ('reflectance', np.float32)))


    def __init__(self, filepath, **header_kwargs):
        """
        Connects to a AWI binary ALS data file. The data is not parsed into memory when this is class is called,
        only the header information that is necessary to decode the binary data structure.

        Usage:
        ======

            alsfile = AirborneLaserScannerFile(filename)
            als = alsfile.get_data(start, stop)

        :param filepath: (str) The path of the AWI ALS file

        :param header_kwargs: Keywords for the header parsing class (
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

        # Init the data output
        als = ALSData(self.line_variables, (nlines, nshots))
        als.set_debug_data(startbyte=startbyte, nbytes=nbytes, line_range=line_range)

        # Read the binary data
        bindat = np.ndarray(shape=(nlines), dtype=object)
        with open(self.filepath, 'rb') as f:
            for i in np.arange(n_selected_lines):
                f.seek(startbyte)
                bindat[i] = f.read(nbytes)
                startbyte = np.uint64(startbyte + nbytes)

        # Unpack the binary data
        # TODO: This is clunky, find a better way
        for i in np.arange(nlines):
            line = bindat[i]
            i0, i1 = 0, 8*nshots
            als.timestamp[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            als.latitude[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            als.longitude[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            als.elevation[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])

        # Filter invalid variables
        if sanitize:
            als.sanitize()

        # All done, return
        return als

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
            # TODO: Use logging
            logging.warning("start time {start} before actual start of file {fstart}".format(start=start, fstart=fstart))
        if stop > fstop:
            logging.warning("stop time {stop} after actual end of file {fstop}".format(stop=stop, fstop=fstop))

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
            np.where(self.line_timestamp <= end_seconds)[0][-1]]
        n_selected_lines = line_range[1] - line_range[0]

        return line_range, n_selected_lines

    def _read_line_timestamp(self):
        """ Read the line time stamp """
        with open(self.filepath, 'rb') as f:
            f.seek(self.header.byte_size)
            data = f.read(self.header.bytes_sec_line)
        struct_def = ">{scan_lines}L".format(scan_lines=self.header.scan_lines)
        self.line_timestamp = np.array(struct.unpack(struct_def, data))


class ALSFileHeader(object):
    """ Class for parsing and storing header information of binary AWI ALS data files """

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

        # Read the header
        with open(filepath, 'rb') as f:

            # Read header size
            self.byte_size = struct.unpack('>b', f.read(1))[0]
            logging.info("als_header.byte_size: %s" % str(self.byte_size))
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
                if key == "device_name" and self.device_name_override is not None:
                    value = self.device_name_override
                setattr(self, key, value)

        # Check if the header was parsed correctly
        self._status = 0
        self._status_context = ""
        try:
            self._validate()
        except:
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


class ALSData(object):
    """ A data class container for ALS data"""

    def __init__(self, vardef, shape):
        """
        Data container for ALS data ordered in scan lines.
        NOTE: Upon initialization this container will be empty. The content must be added directly.
        :param filedef: (dict) Variable definition {varname: dtype, ... }
        :param shape: The shape of the (nlines, nshots) of the data
        """

        # Store arguments
        self.vardef = vardef
        self.shape = shape

        self.debug_data = {}

        # Create the array entries
        for key in vardef.keys():
            setattr(self, key, np.ndarray(shape=shape, dtype=vardef[key]))

    def set_debug_data(self, **kwargs):
        self.debug_data.update(kwargs)

    def sanitize(self):
        """ Run a series of test to identify illegal data points (e.g. out of bounds lon/lat, etc) """

        # Find illegal latitude values
        illegal_lat = np.where(np.abs(self.latitude) > 90.0)
        illegal_lon = np.where(np.abs(self.longitude) > 180.0)
        for illegal_values in [illegal_lat, illegal_lon]:
            for key in self.vardef.keys():
                var = getattr(self, key)
                var[illegal_values] = np.nan
                setattr(self, key, var)

    @property
    def dims(self):
        return self.elevation.shape

    @property
    def n_lines(self):
        return self.dims[0]

    @property
    def n_shots(self):
        return self.dims[1]

    @property
    def lat_range(self):
        return np.nanmin(self.latitude), np.nanmax(self.latitude)

    @property
    def lon_range(self):
        return np.nanmin(self.longitude), np.nanmax(self.longitude)

    @property
    def elev_range(self):
        return np.nanmin(self.elevation), np.nanmax(self.elevation)

