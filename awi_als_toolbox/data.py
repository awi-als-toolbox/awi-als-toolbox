# -*- coding: utf-8 -*-


import numpy as np

from datetime import datetime


class ALSData(object):
    """ A data class container for ALS data"""

    vardef = ["time", "longitude", "latitude", "elevation"]

    def __init__(self, time, lon, lat, elev, segment_window=None):
        """
        Data container for ALS data ordered in scan lines.
        :param time:
        :param lon:
        :param lat:
        :param elev:
        :param segment_window:
        """

        # Add Metadata
        self.metadata = ALSMetadata()
        self.debug_data = {}
        self.segment_window = segment_window

        # Flightdata container (optional)
        self.flightdata = None

        # save data arrays
        # TODO: Validate shapes etc.
        self.time = time
        self.longitude = lon
        self.latitude = lat
        self.elevation = elev

        # Update the metadata now with the data in place
        self._set_metadata()

    def set_debug_data(self, **kwargs):
        self.debug_data.update(kwargs)

    def set_flightdata(self, flightdata):
        """
        Add gps data for the entire flight
        :param flightdata: (FlightGPSData) gps data (time, lon, lat, alt) for full flight
                                           (seconds since the same epoch as laserscanner data)
        :return: None
        """
        self.flightdata = flightdata

    def get_flightdata_segment_subset(self):
        """
        Returns the lon/lat measurement of the flightdata (if present) for the time of the ALS segment
        :return: (longitude, latitude) as numpy arrays or None if flightdata has not been set
        """
        if self.flightdata is None:
            return None
        else:
            return self.flightdata.get_subset(self.tcs_segment_time, self.tce_segment_time)

    def sanitize(self):
        """ Run a series of test to identify illegal data points (e.g. out of bounds lon/lat, etc) """

        # Find illegal latitude values
        illegal_lat = np.where(np.abs(self.latitude) > 90.0)
        illegal_lon = np.where(np.abs(self.longitude) > 180.0)
        for illegal_values in [illegal_lat, illegal_lon]:
            for key in self.vardef:
                var = getattr(self, key)
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
        lon_min, lon_max = self.lat_range
        self.metadata.set_attribute("geospatial_lon_min", lon_min)
        self.metadata.set_attribute("geospatial_lon_max", lon_max)
        elev_min, elev_max = self.elev_range
        self.metadata.set_attribute("geospatial_vertical_min", elev_min)
        self.metadata.set_attribute("geospatial_vertical_max", elev_max)

        # Compute time parameters
        tcs = datetime.utcfromtimestamp(np.nanmin(self.time))
        tce = datetime.utcfromtimestamp(np.nanmax(self.time))
        self.metadata.set_attribute("time_coverage_start", tcs)
        self.metadata.set_attribute("time_coverage_end", tce)

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

    @property
    def has_valid_data(self):
        """ Returns a flag whether the object contains valid elevation data """
        return np.count_nonzero(np.isfinite(self.elevation)) > 0

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
        if self.segment_window is None:
            return np.floor(np.nanmin(self.time))
        else:
            return self.segment_window[0][0]

    @property
    def tce_segment_time(self):
        """
        Return the segment end time in seconds since epoch
        :return: (double) seconds since epoch
        """
        if self.segment_window is None:
            return np.floor(np.nanmax(self.time))
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
        if self.segment_window is None:
            return np.floor(np.nanmin(self.time))
        else:
            return self.segment_window[0][1]


    @property
    def tce_segment_seconds(self):
        """
        Return the segment end time in seconds since start of day
        :return: (double) seconds from ALS file
        """
        if self.segment_window is None:
            return np.floor(np.nanmax(self.time))
        else:
            return self.segment_window[1][1]


class ALSMetadata(object):
    """
    A container for product metadata following CF/Attribute Convention for Data Discovery 1.3
    -> http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery_1-3
    """

    ATTR_DICT = ["title", "summary", "keywords", "Conventions", "id", "naming_authority",
                 "history", "source", "processing_level", "comment", "acknowledgement", "license",
                 "standard_name_vocabulary", "date_created", "creator_name", "creator_url", "creator_email",
                 "institution", "project", "publisher_name", "publisher_url","publisher_email",
                 "geospatial_bound", "geospatial_bounds_crs", "geospatial_bounds_vertical_crs",
                 "geospatial_lat_min",  "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max",
                 "geospatial_vertical_min", "geospatial_vertical_max", "time_coverage_start", "time_coverage_end",
                 "time_coverage_duration", "time_coverage_resolution","creator_type", "creator_institution",
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
        :param attr_dict: (dict) a dict of metadata attributes
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


class FlightGPSData(object):

    def __init__(self, time, lon, lat, alt):
        self.time = time
        self.lon = lon
        self.lat = lat
        self.alt = alt

    def get_lonlats(self):
        """
        Simply return lon, lat
        :return: (longitude, latitudes)
        """
        return self.lon, self.lat

    def get_subset(self, tcs, tce):
        """
        Return a longitude latitude subset
        :param tcs: (double) time coverage start
        :param tce: (double) time coverage end
        :return: FlightGPSData instance with the subset
        """

        tcs_dt = datetime.utcfromtimestamp(tcs)
        tce_dt = datetime.utcfromtimestamp(tce)
        start_dt = datetime.utcfromtimestamp(np.nanmin(self.time))
        end_dt = datetime.utcfromtimestamp(np.nanmax(self.time))

        in_search_window = np.logical_and(self.time >= tcs, self.time <= tce)
        indices = np.where(in_search_window)[0]

        output = FlightGPSData(self.time[indices], self.lon[indices], self.lat[in_search_window],
                               self.alt[indices])
        return output

