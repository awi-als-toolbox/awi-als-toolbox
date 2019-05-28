# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks"


import numpy as np

from collections import OrderedDict

import struct
import logging


class ALSL1BFileDefinition():
    """ TODO: Move this to configuration file """
    def __init__(self):
        self.set_header_info()
        self.set_line_variables()

    def set_line_variables(self):
        self.line_variables = OrderedDict((('timestamp', np.float64),
                                           ('longitude', np.float64),
                                           ('latitude', np.float64),
                                           ('elevation', np.float64),
                                           ('amplitude', np.float32),
                                           ('reflectance', np.float32)))

    def set_header_info(self):
        self.header_dict = OrderedDict((('scan_lines', [4, '>L']),
                                        ('data_points_per_line', [2, '!H']),
                                        ('bytes_per_line', [2, '>H']),
                                        ('bytes_sec_line', [8, '>Q']),
                                        ('year', [2, '>H']),
                                        ('month', [1, '>b']),
                                        ('day', [1, '>b']),
                                        ('start_time_sec', [4, '>L']),
                                        ('stop_time_sec', [4, '>L']),
                                        ('device_name', [8, '>8s'])))


class AirborneLaserScannerFile(object):
    """ Not more than a proof of concept yet """
    def __init__(self):
        self.filename = None
        self.header = ALSL1BHeader()
        self.line_timestamp = None
        self.connected = False
        self.n_selected_lines = 0
        self.filedef = ALSL1BFileDefinition()

    def connect(self, filename):
        """
        Connect to ALS Level1b binary file: retrieve header information
        """
        self.filename = filename
        self.read_header()
        self.read_line_timestamp()
        self.connected = True
        self.set_full_time_range()

    def read_header(self, verbose=True):
        """ Read the header of the ALS level 1b file """
        header_dict = self.filedef.header_dict
        with open(self.filename, 'rb') as f:
            # Read header size
            self.header.byte_size = struct.unpack('>b', f.read(1))[0]
            logging.info("als_header.byte_size: %s" %
                         str(self.header.byte_size))
            if self.header.byte_size == 36:
                header_dict['data_points_per_line'] = [1, '>B']
            elif self.header.byte_size == 37:
                header_dict['data_points_per_line'] = [2, '>H']
            else:
                raise ValueError("Unkown ALS L1B header size:",
                                 self.header.byte_size,
                                 "\nShould be 36 or 37 or unsupported Device")
            # Read Rest of header
            for key in header_dict.keys():
                nbytes, fmt = header_dict[key][0], header_dict[key][1]
                setattr(self.header, key,
                        struct.unpack(fmt, f.read(nbytes))[0])
                if verbose:
                    logging.info("als_header.%s: %s" %
                                 (key, str(getattr(self.header, key))))

    def read_line_timestamp(self):
        """ Read the line time stamp """
        with open(self.filename, 'rb') as f:
            f.seek(self.get_start_byte_linetimestamp())
            data = f.read(self.header.bytes_sec_line)
        struct_def = ">{scan_lines}L".format(scan_lines=self.header.scan_lines)
        self.line_timestamp = np.array(struct.unpack(struct_def, data))

    def read_data(self):
        """  Read lines defined by start and stop second of day """
        self.validate_file()
        startbyte, nbytes = self.get_data_bytes()
#        bindat_dtype = "s%s" % str(nbytes)
        bindat = np.ndarray(shape=(self.n_selected_lines), dtype=object)
        with open(self.filename, 'rb') as f:
            for i in np.arange(self.n_selected_lines):
                f.seek(startbyte)
                bindat[i] = f.read(nbytes)
                startbyte += nbytes
            # bindat = np.fromfile(f, count=nbytes)
        self.set_data_variables()
        self.data = self.unpack_binary_line_data(bindat)

    def unpack_binary_line_data(self, bindat):
        """ TODO: working solution, but needs to be improved """
        nlines = self.n_selected_lines
        nshots = self.header.data_points_per_line
        start_byte, stop_byte = 0, self.header.bytes_per_line
        for i in np.arange(nlines):
            line = bindat[i]
            i0, i1 = 0, 8*nshots
            self.timestamp[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            self.latitude[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            self.longitude[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            start_byte += self.header.bytes_per_line
            stop_byte += self.header.bytes_per_line
            self.elevation[i, :] = struct.unpack(">{n}d".format(n=nshots), line[i0:i1])

    def set_data_variables(self):
        """ Create the numpy arrays for unpacking of binary line data """
        nlines = self.n_selected_lines
        nshots = self.header.data_points_per_line
        for key in self.filedef.line_variables.keys():
            setattr(self, key,
                    np.ndarray(shape=(nlines, nshots),
                               dtype=self.filedef.line_variables[key]))

    def set_time_range(self, time_range):
        """ Sets the first and last line of the subsection """
        self.validate_file()
        self.validate_time_range(time_range[0], time_range[1])
        self.line_index = [
            np.where(self.line_timestamp >= time_range[0])[0][0],
            np.where(self.line_timestamp <= time_range[1])[0][-1]]
        self.n_selected_lines = self.line_index[1] - self.line_index[0]

    def set_full_time_range(self):
        """ Set the full time range as selected content """
        self.validate_file()
        self.line_index = [0, self.header.scan_lines-1]
        self.n_selected_lines = self.header.scan_lines

    def validate_time_range(self, start, stop):
        """ Check for oddities in the time range selection """
        fstart = self.line_timestamp[0]
        fstop = self.line_timestamp[-1]
        # Raise Errors
        if start > stop:
            raise ValueError(
                "start time {start} after stop time {stop}".format(
                    start=start, stop=stop))
        if start > fstop or stop < fstart:
            raise ValueError(
                "time range {start} - {stop} out of bounds " +
                "{fstart} - {fstop}".format(
                    start=start, stop=stop, fstart=fstart, fstop=fstop))
        # Raise Warnings
        if start < fstart:
            # TODO: Use logging
            warnings.warn("start time {start} before actual start of " +
                          "file {fstart}".format(start=start, fstart=fstart))
        if stop > fstop:
            warnings.warn("stop time {stop} after actual end of file " +
                          "{fstop}".format(stop=stop, fstop=fstop))

    def validate_file(self):
        """ Check if file has been specified correctly """
        if not self.connected:
            raise IOError("not connected to file -> self.connect(filename)")

    def get_data_bytes(self):
        """ Returns start and stop bytes of the selected data section """
        # Start byte of scan line
        startbyte = np.uint32(self.header.byte_size)
        startbyte += np.uint32(self.header.bytes_sec_line)
        startbyte += np.uint32(self.line_index[0]) * \
            np.uint32(self.header.bytes_per_line)
        # Number bytes for selected scan lines
        # n_scan_lines = self.line_index[1]-self.line_index[0]
        nbytes = self.header.bytes_per_line  # * n_scan_lines
        return startbyte, nbytes

    def get_start_byte_linetimestamp(self):
        """ Returns the start byte of the line timestamp array """
        return self.header.byte_size

    def get_center_beam_index(self):
        """ Returns index of the center beam """
        if not self.connected:
            return 0
        return np.median(np.arange(self.header.data_points_per_line))

    def get_n_shots_per_line(self):
        """ Returns index of the center beam """
        if not self.connected:
            return 0
        return self.header.data_points_per_line


class ALSL1BHeader(object):

    def __init__(self):
        pass