from abc import ABC, abstractmethod
import os
import glob
import inspect
import time
import json
import traceback
import pymongo
import pytz
import pandas as pd
from numba import jit
import numpy as np
import datetime
from xml.etree import ElementTree
from keras.models import load_model
from PIL import Image, ImageOps


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    """
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    """
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def utc_now():
    return datetime.datetime.now(pytz.utc)


def time_stamps():
    """
    :return: local time, UTC time
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'), \
           datetime.datetime.utcnow().strftime('%Y%m%d_%H:%M:%S')


@jit
def deg2hms(x):
    """Transform degrees to *hours:minutes:seconds* strings.
    Parameters
    ----------
    x : float
        The degree value c [0, 360) to be written as a sexagesimal string.
    Returns
    -------
    out : str
        The input angle written as a sexagesimal string, in the
        form, hours:minutes:seconds.
    """
    assert 0.0 <= x < 360.0, 'Bad RA value in degrees'
    # ac = Angle(x, unit='degree')
    # hms = str(ac.to_string(unit='hour', sep=':', pad=True))
    # print(str(hms))
    _h = np.floor(x * 12.0 / 180.)
    _m = np.floor((x * 12.0 / 180. - _h) * 60.0)
    _s = ((x * 12.0 / 180. - _h) * 60.0 - _m) * 60.0
    hms = '{:02.0f}:{:02.0f}:{:07.4f}'.format(_h, _m, _s)
    # print(hms)
    return hms


@jit
def deg2dms(x):
    """Transform degrees to *degrees:arcminutes:arcseconds* strings.
    Parameters
    ----------
    x : float
        The degree value c [-90, 90] to be converted.
    Returns
    -------
    out : str
        The input angle as a string, written as degrees:minutes:seconds.
    """
    assert -90.0 <= x <= 90.0, 'Bad Dec value in degrees'
    # ac = Angle(x, unit='degree')
    # dms = str(ac.to_string(unit='degree', sep=':', pad=True))
    # print(dms)
    _d = np.floor(abs(x)) * np.sign(x)
    _m = np.floor(np.abs(x - _d) * 60.0)
    _s = np.abs(np.abs(x - _d) * 60.0 - _m) * 60.0
    dms = '{:02.0f}:{:02.0f}:{:06.3f}'.format(_d, _m, _s)
    # print(dms)
    return dms


class Manager(object):

    def __init__(self, _config_file='config.json', _obsdate=None):
        self.__subscribers = set()

        self.config = self.get_config(_config_file)

        # if None, will look for alerts from this night
        self.obsdate = _obsdate
        # base dir to look for data
        self.path_data = self.config['path']['path_data']

        # Check that the directory exists
        if not os.path.exists(self.path_data):
            os.makedirs(self.path_data)

        # keep track of processed stuff
        self.processed = dict()

        print(*time_stamps(), 'MANAGER: AND NOW MY WATCH BEGINS!')

    @staticmethod
    def get_config(_config_file='config.json'):
        """
            load config data in json format
        """
        try:
            ''' script absolute location '''
            abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

            if _config_file[0] not in ('/', '~'):
                if os.path.isfile(os.path.join(abs_path, _config_file)):
                    config_path = os.path.join(abs_path, _config_file)
                else:
                    raise IOError('Failed to find config file')
            else:
                if os.path.isfile(_config_file):
                    config_path = _config_file
                else:
                    raise IOError('Failed to find config file')

            with open(config_path) as cjson:
                config_data = json.load(cjson)
                # config must not be empty:
                if len(config_data) > 0:
                    return config_data
                else:
                    raise Exception('Failed to load config file')

        except Exception as _e:
            print(*time_stamps(), _e)
            raise Exception('Failed to read in the config file')

    def subscribe(self, subscriber):
        self.__subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        self.__subscribers.remove(subscriber)

    def notify(self, message):
        for subscriber in self.__subscribers:
            subscriber.update(message)


class AbstractObserver(ABC):

    def __init__(self, _config_file='config.json'):
        self.config = self.get_config(_config_file)

        # db:
        self.db = None
        self.init_db()
        self.connect_to_db()

        # DL models:
        self.models = dict()
        for model in ('rb', 'sl'):
            print(*time_stamps(), f'loading {model} model')
            self.models[model] = load_model(os.path.join(self.config['path']['path_models'],
                                                         self.config['models'][model]))

        self.model_input_shape = self.models['rb'].input_shape[1:3]

        print(*time_stamps(), 'OBSERVER: AND NOW MY WATCH BEGINS!')

    @staticmethod
    def get_config(_config_file='config.json'):
        """
            load config data in json format
        """
        try:
            ''' script absolute location '''
            abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

            if _config_file[0] not in ('/', '~'):
                if os.path.isfile(os.path.join(abs_path, _config_file)):
                    config_path = os.path.join(abs_path, _config_file)
                else:
                    raise IOError('Failed to find config file')
            else:
                if os.path.isfile(_config_file):
                    config_path = _config_file
                else:
                    raise IOError('Failed to find config file')

            with open(config_path) as cjson:
                config_data = json.load(cjson)
                # config must not be empty:
                if len(config_data) > 0:
                    return config_data
                else:
                    raise Exception('Failed to load config file')

        except Exception as _e:
            print(*time_stamps(), _e)
            raise Exception('Failed to read in the config file')

    def init_db(self):
        _client = pymongo.MongoClient(username=self.config['database']['admin'],
                                      password=self.config['database']['admin_pwd'],
                                      host=self.config['database']['host'],
                                      port=self.config['database']['port'])
        # _id: db_name.user_name
        user_ids = [_u['_id'] for _u in _client.admin.system.users.find({}, {'_id': 1})]

        db_name = self.config['database']['db']
        username = self.config['database']['user']

        # print(f'{db_name}.{username}')
        # print(user_ids)

        if f'{db_name}.{username}' not in user_ids:
            _client[db_name].command('createUser', self.config['database']['user'],
                                     pwd=self.config['database']['pwd'], roles=['readWrite'])
            print(*time_stamps(), 'Successfully initialized db')

    def connect_to_db(self):
        """
            Connect to database
        :return:
        """

        _config = self.config

        try:
            # there's only one instance of DB, it's too big to be replicated
            _client = pymongo.MongoClient(host=_config['database']['host'],
                                          port=_config['database']['port'], connect=False)
            # grab main database:
            _db = _client[_config['database']['db']]
        except Exception as _e:
            raise ConnectionRefusedError
        try:
            # authenticate
            _db.authenticate(_config['database']['user'], _config['database']['pwd'])
        except Exception as _e:
            raise ConnectionRefusedError

        self.db = dict()
        self.db['client'] = _client
        self.db['db'] = _db

        print(*time_stamps(), "Connected to db")

    def insert_db_entry(self, _collection=None, _db_entry=None):
        """
            Insert a document _doc to collection _collection in DB.
            It is monitored for timeout in case DB connection hangs for some reason
        :param _collection:
        :param _db_entry:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _db_entry is not None, 'Must specify document'
        try:
            self.db['db'][_collection].insert_one(_db_entry)
        except Exception as _e:
            print(*time_stamps(), 'Error inserting {:s} into {:s}'.format(str(_db_entry['_id']), _collection))
            traceback.print_exc()
            print(_e)

    def insert_multiple_db_entries(self, _collection=None, _db_entries=None):
        """
            Insert a document _doc to collection _collection in DB.
            It is monitored for timeout in case DB connection hangs for some reason
        :param _db:
        :param _collection:
        :param _db_entries:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _db_entries is not None, 'Must specify documents'
        try:
            # ordered=False ensures that every insert operation will be attempted
            # so that if, e.g., a document already exists, it will be simply skipped
            self.db['db'][_collection].insert_many(_db_entries, ordered=False)
        except pymongo.errors.BulkWriteError as bwe:
            print(*time_stamps(), bwe.details)
        except Exception as _e:
            traceback.print_exc()
            print(_e)

    def replace_db_entry(self, _collection=None, _filter=None, _db_entry=None):
        """
            Insert a document _doc to collection _collection in DB.
            It is monitored for timeout in case DB connection hangs for some reason
        :param _collection:
        :param _filter:
        :param _db_entry:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _db_entry is not None, 'Must specify document'
        try:
            self.db['db'][_collection].replace_one(_filter, _db_entry, upsert=True)
        except Exception as _e:
            print(*time_stamps(), 'Error replacing {:s} in {:s}'.format(str(_db_entry['_id']), _collection))
            traceback.print_exc()
            print(_e)

    @abstractmethod
    def update(self, message):
        pass


class Watcher(AbstractObserver):

    def update(self, message):
        pass

