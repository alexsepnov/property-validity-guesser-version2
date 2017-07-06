# -*- coding: utf-8 -*-
"""
.. module:: preprocessor
.. moduleauthor:: Alex Sepnov <alex@urbanindo.com>
"""
import sys
import math
import csv
import re
import numpy as np
import pickle

csv.field_size_limit(sys.maxsize)

csv.register_dialect(
    'csv_dialect',
    delimiter=',',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)

csv.register_dialect(
    'tsv_dialect',
    delimiter='\t',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)

WHITELIST_INDEX = 0.5
BANNED_WORD_FOR_ADDRESS = ["provinsi", "kota", "provinsi", "kotamadya",
                       "kabupaten", "kelurahan", "kecamatan", "desa"]
BANNED_WORD_FOR_ADDITIONAL_REGION = ["jalan", "jln", "jl"]

##==============================================================================
class LocationBlacklist(object):
    whitelist_index = WHITELIST_INDEX

    def __init__(self):
        '''
        '''
        self.dictionary_blacklist = {}

    def process_file(self, filename):
        '''
        :param filename:
        row[0] = longitude
        row[1] = latitude
        row[3] = blacklist index, 0 = whitelist, 2 = blacklist
        '''
        coord_b = []
        index_b = []

        with open(filename) as csvfile:
            reader = csv.reader(csvfile, dialect='csv_dialect')

            for row in reader:
                coord = (row[0], row[1])
                coord_b.append(coord)
                index_b.append(int(row[2]))

        self.dictionary_blacklist.update(zip(coord_b, index_b))

    def is_blacklisted(self, longitude, latitude):
        '''
        :param longitude:
        :param latitude:
        :return: blacklist index, 0 = whitelist, 1 = greylist, 2 = blacklist
        '''
        coordinate = (longitude, latitude)

        return float(coordinate in self.dictionary_blacklist) \
            if coordinate in self.dictionary_blacklist \
            else LocationBlacklist.whitelist_index

    def write_to_file(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load_from_file(filename):
        location_blacklist_object = pickle.load(open(filename, "rb"))
        return location_blacklist_object

##==============================================================================
class PreprocessorNormalizer(object):
    banned_word_for_address = BANNED_WORD_FOR_ADDRESS
    banned_word_for_additional_region = BANNED_WORD_FOR_ADDITIONAL_REGION

    @staticmethod
    def normalize_address_length(val):
        '''
        :param val:address length
        :return: log value of address length
        '''
        return math.log10(min(100, int(val)) + 1)

    @staticmethod
    def string_norm_regex(string):
        '''
        to strip down string into lowercase + removing excessive space
        :param any string related to location (address or additional region)
        :return: lowercase string without excessive space
        '''
        return " ".join(string.strip().lower().split())

    @staticmethod
    def normalize_commas_length(string):
        '''
        :param string: address string
        :return: log-normalized value of number of commas
        '''
        return math.log(min(9, len(string.split(','))), 3)

    @staticmethod
    def address_contains_banned_words(string):
        '''
        :param string: address string
        :return: 1/0 (boolean) if string contains banned word or not
        '''
        string_array = re.split(', |; | ,| ;|,|;|\s', string)

        for str in string_array:
            if str in PreprocessorLearning.banned_word_for_address:
                return 1

        return 0

    @staticmethod
    def additional_region_contains_banned_words(string):
        '''
        :param string: additional region
        :return: 1/0 (boolean) if string contains banned word or not
        '''
        string_array = re.split(', |; |\. | ,| ;| \.|,|;|\.|\s', string)

        for str in string_array:
            if str in PreprocessorLearning.banned_word_for_address:
                return 1

        return 0

    @staticmethod
    def user_wrongness_value(wrong_properties, reviewed_properties):
        '''
        :param wrong_properties: number of wrong properties of a user
        :param reviewed_properties: number of reviewed properties of a user
        :return: ratio of (wrong_properties/reviewed properties)
        '''
        user_reviewed_properties = float(reviewed_properties)
        return float(wrong_properties)/user_reviewed_properties if user_reviewed_properties > 0.0 else 0.0

##==============================================================================
class PreprocessorLearning(PreprocessorNormalizer):
    move_distance_limit = 200.0

    def __init__(self):
        self.x = []
        self.y = []
        self.location_blacklist = None

    def load_location_blacklist(self, location_blacklist_object):
        self.location_blacklist = location_blacklist_object

    def process_file(self, filename):
        '''
        :param filename: address file name for learning
                         format--> see "process_row" method
        '''
        assert self.location_blacklist is not None

        with open(filename) as csvfile:
            reader = csv.reader(csvfile, dialect='tsv_dialect')
            next(reader, None) # skip the headers

            for row in reader:
                xt, yt = self.process_row(row)
                self.x.append(xt)
                self.y.append(yt)

    def get_data(self):
        '''
        :return: numpy array of 1. x (feature data)
                                2. y (label data)
        '''
        return np.array(self.x), np.array(self.y)

    def process_row(self, row):
        '''
        :param row:
        row:0	propertyId
        row:1	longitudeBefore
        row:2	latitudeBefore
        row:3	addressBefore
        row:4	addressBeforeCharLength
        row:5	addressBeforeContainsJalan
        row:6	additionalRegionBefore
        row:7	additionalRegionBeforeCharLength
        row:8	isVenueGrouped
        row:9	userPropertyCount
        row:10	userWrongProperty
        row:11	userReviewedProperty
        row:12	userCheckedProperty
        row:13	longitudeAfter
        row:14	latitudeAfter
        row:15	addressAfter
        row:16	addressAfterCharLength
        row:17	addressAfterContainsJalan
        row:18	additionalRegionAfter
        row:19	additionalRegionAfterCharLength
        :return: x (feature data)
                 y (label data)
        '''

        x = []
        # total learning features = 9

        #normalized some strings first
        address = PreprocessorLearning.string_norm_regex(row[3])
        additional_region = PreprocessorLearning.string_norm_regex(row[6])

        #x0 = address' character length
        x.append(PreprocessorLearning.normalize_address_length(row[4]))

        #x1 = commas' length in address
        x.append(PreprocessorLearning.normalize_commas_length(address))

        #x2 = if address contains "jalan"
        x.append(1 if row[5] == 'no' else 0)

        #x3 = if address contains banned words
        x.append(PreprocessorLearning.address_contains_banned_words(address))

        #x4 = if additional region contains banned words
        x.append(PreprocessorLearning.additional_region_contains_banned_words
                 (additional_region))

        #x5 = user wrongness
        x.append(PreprocessorLearning.user_wrongness_value(row[10], row[11]))

        #x6 = is venue grouped?
        x.append(1 if row[8] == 'no' else 0)

        #x7  = if additional region is a duplicate of address
        x.append(1 if address == additional_region else 0)

        #x8  = if additional region available
        x.append(1 if int(row[7]) <= 0 else 0)

        #x9 = if blacklisted, whitelist=0, blacklist=2,
        #                     greylist=see default value in LocationBlacklist
        x.append(self.location_blacklist.is_blacklisted(row[1], row[2]))

        y = 0 #predicted as correct
        coordinate_before = (float(row[1]), float(row[2]))
        coordinate_after = (float(row[13]), float(row[14]))
        move_distance = PreprocessorLearning.haversine_distance(coordinate_before,
                                                                coordinate_after)
        #conditional if to be predicted as incorrect (=1)
        if (move_distance > PreprocessorLearning.move_distance_limit) or \
           (row[8] == 'no') or (address == additional_region) or \
           (int(row[7]) <= 0):
            y = 1

        return x, y

    @staticmethod
    def haversine_distance(coordinate_before, coordinate_after):
        '''
        :param coordinate_before: tuple in format (longitude, latitude)
        :param coordinate_after: tuple in format (longitude, latitude)
        :return: distance between 2 coordinates in meters
        '''
        longitude1, latitude1 = coordinate_before
        longitude2, latitude2 = coordinate_after
        radius = 6371000  #in meter

        dlatitude = math.radians(latitude2 - latitude1)
        dlongitude = math.radians(longitude2 - longitude1)
        a = math.sin(dlatitude / 2.0) ** 2 + \
            math.cos(math.radians(latitude1)) * \
            math.cos(math.radians(latitude2)) * \
            math.sin(dlongitude / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return radius * c

##==============================================================================
class PreprocessorRow(PreprocessorNormalizer):
    def __init__(self):
        self.location_blacklist = None

    def load_location_blacklist(self, location_blacklist_object):
        self.location_blacklist = location_blacklist_object

    def process_row(self, row):
        '''
        :param row:
        row:0	propertyId
        row:1	longitudeBefore
        row:2	latitudeBefore
        row:3	addressBefore
        row:4	addressBeforeCharLength
        row:5	addressBeforeContainsJalan
        row:6	additionalRegionBefore
        row:7	additionalRegionBeforeCharLength
        row:8	isVenueGrouped
        row:9	userPropertyCount
        row:10	userWrongProperty
        row:11	userReviewedProperty
        row:12	userCheckedProperty
        :return: 1. property id,
                 2. x (feature data)
        '''
        property_id = int(row[0])

        x = []
        # total learning features = 9

        # normalized some strings first
        address = PreprocessorLearning.string_norm_regex(row[3])
        additional_region = PreprocessorLearning.string_norm_regex(row[6])

        # x0 = address' character length
        x.append(PreprocessorLearning.normalize_address_length(row[4]))

        # x1 = commas' length in address
        x.append(PreprocessorLearning.normalize_commas_length(address))

        # x2 = if address contains "jalan"
        x.append(1 if row[5] == 'no' else 0)

        # x3 = if address contains banned words
        x.append(PreprocessorLearning.address_contains_banned_words(address))

        # x4 = if additional region contains banned words
        x.append(PreprocessorLearning.additional_region_contains_banned_words
                 (additional_region))

        # x5 = user wrongness
        x.append(PreprocessorLearning.user_wrongness_value(row[10], row[11]))

        # x6 = is venue grouped?
        x.append(1 if row[8] == 'no' else 0)

        # x7  = if additional region is a duplicate of address
        x.append(1 if address == additional_region else 0)

        # x8  = if additional region available
        x.append(1 if int(row[7]) <= 0 else 0)

        # x9 = if blacklisted, whitelist=0, blacklist=2,
        #                     greylist=see default value in LocationBlacklist
        x.append(self.location_blacklist.is_blacklisted(row[1], row[2]))

        return property_id, x
