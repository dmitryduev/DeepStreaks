from abc import ABC, abstractmethod
import argparse
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
from copy import deepcopy


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

    def __init__(self, _config_file='config.json', _obsdate=None, _enforce=False):
        self.__subscribers = set()

        self.config = get_config(_config_file)

        self.enforce = _enforce

        # if None, will look for alerts from this night
        self.obsdate = _obsdate
        # base dir to look for data
        self.path_data = self.config['path']['path_data']

        # Check that the directory exists
        if not os.path.exists(self.path_data):
            os.makedirs(self.path_data)

        # keep track of processed stuff
        self.processed_meta = dict()
        self.processed_img = dict()

        print(*time_stamps(), 'MANAGER: AND NOW MY WATCH BEGINS!')

    def subscribe(self, subscriber):
        self.__subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        self.__subscribers.remove(subscriber)

    def notify(self, message):
        for subscriber in self.__subscribers:
            subscriber.update(message)

    @staticmethod
    def find_files(root_dir, extension: str = 'jpg', top_level_only: bool=True):
        for dir_name, subdir_list, file_list in os.walk(root_dir, followlinks=True):
            for f_name in file_list:
                if f_name.endswith(f'.{extension}'):
                    yield os.path.join(dir_name, f_name)
            if top_level_only:
                break

    def run(self):
        """
            This could be replaced with a Kafka watcher in the future
        :return:
        """
        while True:
            if self.enforce or (datetime.datetime.utcnow().hour < 15):

                try:
                    # and now my watch begins
                    if self.obsdate is not None:
                        # looking at particular date?
                        obsdate = self.obsdate
                    else:
                        obsdate = datetime.datetime.utcnow().strftime('%Y%m%d')

                    print(*time_stamps(), f'Processing data from {obsdate}')

                    # clean up self.processed_alerts
                    obsdates = list(self.processed_meta.keys())

                    print(*time_stamps(), 'Dates on watch:', obsdates)

                    if self.obsdate is None:  # only do this if not looking at particular date?
                        for _od in obsdates:
                            if _od != obsdate:
                                print(*time_stamps(), f'No need to look at {_od}, dropping')
                                try:
                                    self.processed_meta.pop(_od, None)
                                    self.processed_img.pop(_od, None)
                                finally:
                                    pass

                    if obsdate not in obsdates:
                        # use set/dict as search operation is much faster
                        self.processed_meta[obsdate] = set()
                        self.processed_img[obsdate] = set()

                    print(*time_stamps(),
                          f'Processed cutout files for {obsdate} so far: {len(self.processed_img[obsdate])}')
                    print(*time_stamps(),
                          f'Processed meta files for {obsdate} so far: {len(self.processed_meta[obsdate])}')

                    # go img cutouts
                    # todo
                    image_files = set(self.find_files(os.path.join(self.path_data, 'stamps', f'stamps_{obsdate}')))
                    # unprocessed files are the difference between two sets
                    unprocessed_image_files = image_files - self.processed_img[obsdate]

                    # todo: magic
                    num_img_files = len(image_files)
                    print(*time_stamps(), f'Found {num_img_files} image cutout files for {obsdate}')

                    if len(unprocessed_image_files) == 0:
                        print(*time_stamps(), f'Apparently already looked at all available image files for {obsdate}')

                    else:
                        # notify subscribed img watcher(s):
                        self.notify(message={'datatype': 'img',
                                             'obsdate': obsdate,
                                             'path_images': unprocessed_image_files})

                    # add (now) processed files to processed_img set
                    # self.processed_img[obsdate] = self.processed_img[obsdate].union(unprocessed_image_files)
                    self.processed_img[obsdate] |= unprocessed_image_files

                    # go meta
                    meta_files = glob.glob(os.path.join(self.path_data, 'meta', obsdate, 'ztf_*_streaks.txt'))
                    num_meta_files = len(meta_files)
                    print(*time_stamps(), f'Found {num_meta_files} meta files for {obsdate}')

                    if len(self.processed_meta[obsdate]) == num_meta_files:
                        print(*time_stamps(), f'Apparently already looked at all available meta files for {obsdate}')

                    else:

                        for fi, filename in enumerate(meta_files):
                            try:
                                print(*time_stamps(), f'{obsdate}', f'{fi+1}/{num_meta_files}',
                                      f'processing {filename}')

                                # strip file name:
                                meta_name = os.path.basename(filename)

                                if meta_name not in self.processed_meta[obsdate]:
                                    # notify subscribed meta watcher(s):
                                    self.notify(message={'datatype': 'meta',
                                                         'obsdate': obsdate,
                                                         'filename': filename})

                                    # save as processed_meta
                                    self.processed_meta[obsdate].add(meta_name)

                                else:
                                    print(*time_stamps(), f'{obsdate}', f'{fi+1}/{num_meta_files}',
                                          f'{filename} already checked, skipping')

                            except Exception as _e:
                                traceback.print_exc()
                                print(*time_stamps(), str(_e))
                                try:
                                    with open(os.path.join(self.path_data, 'issues.log'), 'a+') as f_issues:
                                        _issue = '{:s} {:s} {:s}\n'.format(*time_stamps(), str(_e))
                                        f_issues.write(_issue)
                                finally:
                                    pass

                                continue

                    print(*time_stamps(), f'Done. Processed meta files for {obsdate} so far:',
                          len(self.processed_meta[obsdate]))
                    # take a nap when done
                    print(*time_stamps(), 'Sleeping for 1 minute...')
                    time.sleep(60 * 1)

                except Exception as e:
                    traceback.print_exc()
                    print(*time_stamps(), str(e))
                    print(*time_stamps(), 'Error encountered. Sleeping for 5 minutes...')
                    time.sleep(60 * 5)

            else:
                print(*time_stamps(), 'Sleeping before my watch starts tonight...')
                time.sleep(60 * 5)


class AbstractObserver(ABC):

    def __init__(self, _config_file='config.json', _verbose=True):
        self.config = get_config(_config_file)
        self.verbose = _verbose

        # base dir to look for data
        self.path_data = self.config['path']['path_data']

        # db:
        self.db = None
        self.init_db()
        self.connect_to_db()

        if self.verbose:
            print(*time_stamps(), 'Creating/checking indices')
        self.db['db'][self.config['database']['collection_main']].create_index([('jd', pymongo.DESCENDING)],
                                                                               background=True)
        # index default model scores
        for model in self.config['default_models']:
            self.db['db'][self.config['database']['collection_main']].create_index([(model, pymongo.DESCENDING)],
                                                                                   background=True)
        # index working model scores
        for model in self.config['models']:
            self.db['db'][self.config['database']['collection_main']].create_index([(model, pymongo.DESCENDING)],
                                                                                   background=True)

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

    def insert_or_replace_db_entry(self, _collection=None, _db_entry=None):
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
            self.insert_db_entry(_collection, _db_entry)
        except Exception as _e:
            try:
                if self.verbose:
                    print(*time_stamps(), 'Found entry, updating..')

                # merge scores:
                scores = self.db['db'][_collection].find_one({'_id': _db_entry['_id']},
                                                             {'_id': 0, 'scores': 1})['scores']
                _db_entry_megred_scores = deepcopy(_db_entry)
                new_scores = _db_entry_megred_scores['scores']
                for _model in new_scores:
                    if _model in scores:
                        for _m in new_scores[_model]:
                            scores[_model][_m] = new_scores[_model][_m]
                    else:
                        scores[_model] = new_scores[_model]

                self.replace_db_entry(_collection, {'_id': _db_entry['_id']}, _db_entry_megred_scores)
            except Exception as __e:
                print(*time_stamps(), 'Error inserting/replacing {:s} into {:s}'.format(str(_db_entry['_id']),
                                                                                        _collection))
                traceback.print_exc()
                print(__e)

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

    def update_db_entry(self, _collection=None, _filter=None, _db_entry_upd=None,
                        _upsert=True, _bypass_document_validation=False):
        """
            Update/upsert a document matching _filter to collection _collection in DB.
        :param _collection:
        :param _filter:
        :param _db_entry_upd:
        :param _upsert:
        :param _bypass_document_validation:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _filter is not None, 'Must specify filter'
        assert _db_entry_upd is not None, 'Must specify update statement'
        try:
            self.db['db'][_collection].update_one(_filter, _db_entry_upd,
                                                  upsert=_upsert,
                                                  bypass_document_validation=_bypass_document_validation)
        except Exception as _e:
            print(*time_stamps(), f'Error: updating/upserting in {_collection}: {_filter}  {_db_entry_upd}')
            traceback.print_exc()
            print(_e)

    @abstractmethod
    def update(self, message):
        pass


class WatcherMeta(AbstractObserver):

    def __init__(self, _config_file='config.json', _verbose=True):
        super().__init__(_config_file=_config_file, _verbose=_verbose)

        if self.verbose:
            print(*time_stamps(), 'META OBSERVER: AND NOW MY WATCH BEGINS!')

    def update(self, message):

        datatype = message['datatype'] if 'datatype' in message else None
        assert datatype is not None, (*time_stamps(), 'Bad message: no datatype.')

        if datatype == 'meta':

            filename = message['filename'] if 'filename' in message else None
            assert filename is not None, (*time_stamps(), 'Bad message: no filename.')

            # base_name = filename.split('_streaks.txt')[0]

            obsdate = message['obsdate'] if 'obsdate' in message else None
            assert obsdate is not None, (*time_stamps(), 'Bad message: no obsdate.')

            # digest
            df = pd.read_table(filename, sep='|', header=0, skipfooter=1, engine='python')
            df = df.drop(0)
            for index, row in df.iterrows():
                try:
                    _tmp = row.to_dict()
                    doc = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in _tmp.items()}
                    # manually fix types
                    if 'jd' in doc:
                        doc['jd'] = float(doc['jd'])
                    if 'pid' in doc:
                        doc['pid'] = int(doc['pid'])
                    if 'streakid' in doc:
                        doc['streakid'] = int(doc['streakid'])
                    if 'strid' in doc:
                        doc['strid'] = int(doc['strid'])

                    # doc['_id'] = f'strkid{doc["streakid"]}_pid{doc["pid"]}'
                    doc_id = f'strkid{doc["streakid"]}_pid{doc["pid"]}'

                    # doc['base_name'] = base_name

                    # parse ADES:
                    path_streak = os.path.join(self.path_data, 'stamps', f'stamps_{obsdate}')
                    # path_streak = os.path.join(self.path_data, 'stamps',
                    #                            f'stamps_{obsdate}', f'{base_name}_strkcutouts')
                    path_streak_ades = os.path.join(path_streak, f'{doc["_id"]}_ades.xml')
                    path_streak_stamp = os.path.join(path_streak, f'{doc["_id"]}_scimref.jpg')

                    tree = ElementTree.parse(path_streak_ades)
                    root = tree.getroot()
                    xmldict = XmlDictConfig(root)
                    # print(xmldict)
                    doc['ades'] = xmldict

                    doc['last_modified'] = utc_now()

                    # print(doc)

                    # todo: replace with upsert
                    # self.insert_or_replace_db_entry(_collection=self.config['database']['collection_main'],
                    #                                 _db_entry=doc)
                    self.update_db_entry(_collection=self.config['database']['collection_main'],
                                         _filter={'_id': doc_id}, _db_entry_upd={'$set': doc},
                                         _upsert=True)

                    if self.verbose:
                        print(*time_stamps(), f'Successfully processed {doc["_id"]}.')

                except Exception as _e:
                    traceback.print_exc()
                    print(_e)


class WatcherImg(AbstractObserver):

    def __init__(self, _config_file='config.json', _verbose=True):
        super().__init__(_config_file=_config_file, _verbose=_verbose)

        # DL models:
        self.models = dict()
        for model in self.config['models']:
            if self.verbose:
                print(*time_stamps(), f'loading model {model}: {self.config["models"][model]}')
            self.models[model] = load_model(os.path.join(self.config['path']['path_models'],
                                                         self.config['models'][model]))

        self.model_input_shape = self.models[self.config['default_models']['rb']].input_shape[1:3]

        if self.verbose:
            print(*time_stamps(), 'IMAGE OBSERVER: AND NOW MY WATCH BEGINS!')

    @staticmethod
    def load_data_predict(path_images=(), grayscale: bool = True, resize: tuple = (144, 144)):

        num_images = len(path_images)
        num_channels = 1 if grayscale else 3

        # allocate:
        data = np.zeros((num_images, *resize, num_channels))
        img_ids = np.zeros(num_images, dtype=object)

        for ii, path_image in enumerate(path_images):
            image_basename = os.path.basename(path_image)
            img_id = image_basename.split('_scimref.jpg')[0]
            img_ids[ii] = img_id

            if grayscale:
                img = np.array(ImageOps.grayscale(Image.open(path_image)).resize(resize, Image.BILINEAR)) / 255.
                img = np.expand_dims(img, 2)
            else:
                img = ImageOps.grayscale(Image.open(path_image)).resize(resize, Image.BILINEAR)
                rgbimg = Image.new("RGB", img.size)
                rgbimg.paste(img)
                img = np.array(rgbimg) / 255.

            data[ii, :] = img

        return data, img_ids

    def update(self, message):
        datatype = message['datatype'] if 'datatype' in message else None
        assert datatype is not None, (*time_stamps(), 'Bad message: no datatype.')

        if datatype == 'img':

            path_images = message['path_images'] if 'path_images' in message else None
            assert path_images is not None, (*time_stamps(), 'Bad message: no path_images.')

            # base_name = filename.split('_streaks.txt')[0]

            obsdate = message['obsdate'] if 'obsdate' in message else None
            assert obsdate is not None, (*time_stamps(), 'Bad message: no obsdate.')

            # TODO: digest
            if self.verbose:
                print(*time_stamps(), 'loading image data')
                tic = time.time()
            images, image_ids = self.load_data_predict(path_images=path_images)
            if self.verbose:
                toc = time.time()
                print(*time_stamps(), images.shape)
                print(*time_stamps(), f'done. loaded {len(image_ids)} images, which took {toc-tic} seconds.')

            batch_size = 32

            scores = dict()
            for model in self.config['models']:
                tic = time.time()
                scores[model] = self.models[model].predict(images, batch_size=batch_size)
                toc = time.time()
                if self.verbose:
                    print(*time_stamps(),
                          f'{model}: forward prop with batch_size={batch_size} took {toc-tic} seconds.')
                    print(*time_stamps(), scores[model].shape)

            for ii, image_id in enumerate(image_ids):
                # build doc to upsert into bd:
                doc = dict()
                # doc_models = {model: float(scores[model][ii]) for model in self.config['models']}

                # default DL models
                for dl in self.config['default_models']:
                    doc[dl] = float(scores[self.config['default_models'][dl]][ii])

                # current working models, for the ease of db access:
                for model in self.models:
                    doc[model] = float(scores[model][ii])

                # book-keeping for the future [if a model is retrained]
                doc['scores'] = dict()
                for model in self.models:
                    doc['scores'][model] = {self.config['models'][model].split('.')[0]: float(scores[model][ii])}

                doc['last_modified'] = utc_now()

                self.update_db_entry(_collection=self.config['database']['collection_main'],
                                     _filter={'_id': image_id}, _db_entry_upd={'$set': doc},
                                     _upsert=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Digest ZTF nightly streak data')
    parser.add_argument('--obsdate', help='observing date')
    parser.add_argument('--enforce', action='store_true', help='enforce execution')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    manager = Manager(_config_file=args.config_file, _obsdate=args.obsdate, _enforce=args.enforce)
    watcher_meta = WatcherMeta(_config_file=args.config_file)
    watcher_img = WatcherImg(_config_file=args.config_file)

    manager.subscribe(watcher_meta)
    manager.subscribe(watcher_img)
    manager.run()

    # python watcher.py config.json --obsdate 20180927 --enforce
