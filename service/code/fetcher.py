import os
import datetime
import glob
import shutil
import tarfile
from argparse import ArgumentParser
import requests
from bs4 import BeautifulSoup
import time
import pytz
import inspect
import json
import numpy as np
import pymongo
from numba import jit
import traceback
import logging


''' load config and secrets '''
with open('/app/config.json') as cjson:
    config = json.load(cjson)

with open('/app/secrets.json') as sjson:
    secrets = json.load(sjson)


def utc_now():
    return datetime.datetime.now(pytz.utc)


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


class qaWatcher:
    def __init__(self, obsdate=None, transfer_to_local=True,
                 data_dir='/data/streaks/',
                 batch_size=128):
        """
        Look for new qa files
        """

        # keep config data:
        self.config = config

        ''' set up logging at init '''
        self.logger, self.logger_utc_date = self.set_up_logging(_name='fetcher', _mode='a')
        # print(self.logger, self.logger_utc_date)

        self.base_url = 'https://ztfweb.ipac.caltech.edu/ztf/depot/'
        if obsdate is not None:
            self.obsdate = obsdate
        else:
            self.obsdate = datetime.datetime.utcnow().strftime('%Y%m%d')

        self.stamps_dir = os.path.join(data_dir, 'stamps', f'stamps_{self.obsdate}')
        self.meta_dir = os.path.join(data_dir, 'meta', self.obsdate)

        # Check that the directory exists
        if not os.path.exists(self.stamps_dir):
            os.makedirs(self.stamps_dir)
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

        self.night_url = os.path.join(self.base_url, self.obsdate)

        self.sub_dir_list = []
        self.processed_links = []

        self.processed_stamps = [os.path.basename(x)
                                 for x in glob.glob(os.path.join(self.stamps_dir, '*strkcutouts*gz'))]
        self.processed_meta = [os.path.basename(x)
                               for x in glob.glob(os.path.join(self.meta_dir, '*streaks*'))]

        self.transfer_to_local = transfer_to_local

        # ''' db stuff '''
        # # number of records to insert to db
        # self.batch_size = batch_size
        # self.documents = []
        #
        # # mongo collection name:
        # self.collection_alerts = 'ZTF_alerts'
        # self.db = None
        # self.connect_to_db()

        self.logger.debug('init successful')

    def set_up_logging(self, _name='fetcher', _mode='a'):
        """ Set up logging

            :param _name:
            :param _level: DEBUG, INFO, etc.
            :param _mode: overwrite log-file or append: w or a
            :return: logger instance
            """
        # 'debug', 'info', 'warning', 'error', or 'critical'
        if self.config['misc']['logging_level'] == 'debug':
            _level = logging.DEBUG
        elif self.config['misc']['logging_level'] == 'info':
            _level = logging.INFO
        elif self.config['misc']['logging_level'] == 'warning':
            _level = logging.WARNING
        elif self.config['misc']['logging_level'] == 'error':
            _level = logging.ERROR
        elif self.config['misc']['logging_level'] == 'critical':
            _level = logging.CRITICAL
        else:
            raise ValueError('Config file error: logging level must be ' +
                             '\'debug\', \'info\', \'warning\', \'error\', or \'critical\'')

        # get path to logs from config:
        _path = self.config['path']['path_logs']

        if not os.path.exists(_path):
            os.makedirs(_path)
        utc_now = datetime.datetime.utcnow()

        # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
        _logger = logging.getLogger(_name)

        _logger.setLevel(_level)
        # create the logging file handler
        fh = logging.FileHandler(os.path.join(_path, '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                                 mode=_mode)
        logging.Formatter.converter = time.gmtime

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        _logger.addHandler(fh)

        return _logger, utc_now.strftime('%Y%m%d')

    def shut_down_logger(self):
        """
            Prevent writing to multiple log-files after 'manual rollover'
        :return:
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def check_logging(self):
        """
            Check if a new log file needs to be started and start it if necessary
        """
        if datetime.datetime.utcnow().strftime('%Y%m%d') != self.logger_utc_date:
            # reset
            self.shut_down_logger()
            self.logger, self.logger_utc_date = self.set_up_logging(_name='alert_watcher', _mode='a')

    def connect_to_db(self):
        """
            Connect to Robo-AO's MongoDB-powered database
        :return:
        """
        _config = self.config

        try:
            if self.logger is not None:
                self.logger.debug('Connecting to the database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            # there's only one instance of DB, it's too big to be replicated
            _client = pymongo.MongoClient(host=_config['database']['host'],
                                          port=_config['database']['port'], connect=False)
            # grab main database:
            _db = _client[_config['database']['db']]
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to connect to the database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            raise ConnectionRefusedError
        try:
            # authenticate
            _db.authenticate(_config['database']['user'], _config['database']['pwd'])
            if self.logger is not None:
                self.logger.debug('Successfully authenticated with the database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Authentication failed for the database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            raise ConnectionRefusedError

        # (re)define self.db
        self.db = dict()
        self.db['client'] = _client
        self.db['db'] = _db

    def disconnect_from_db(self):
        """
            Disconnect from MongoDB database.
        :return:
        """
        self.logger.debug('Disconnecting from the database.')
        if self.db is not None:
            try:
                self.db['client'].close()
                self.logger.debug('Successfully disconnected from the database.')
            except Exception as e:
                self.logger.error('Failed to disconnect from the database.')
                self.logger.error(e)
            finally:
                # reset
                self.db = None
        else:
            self.logger.debug('No connection found.')

    # @timeout_decorator.timeout(60, use_signals=False)
    def check_db_connection(self):
        """
            Check if DB connection is alive/established.
        :return: True if connection is OK
        """
        self.logger.debug('Checking database connection.')
        if self.db is None:
            try:
                self.connect_to_db()
            except Exception as e:
                print('Lost database connection.')
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False
        else:
            try:
                # force connection on a request as the connect=True parameter of MongoClient seems
                # to be useless here
                self.db['client'].server_info()
            except pymongo.errors.ServerSelectionTimeoutError as e:
                print('Lost database connection.')
                self.db = None
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False

        return True

    # @timeout_decorator.timeout(60, use_signals=False)
    # @timeout(seconds_before_timeout=60)
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
            print('Error inserting {:s} into {:s}'.format(str(_db_entry['_id']), _collection))
            traceback.print_exc()
            print(_e)

    # @timeout(seconds_before_timeout=60)
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
            print(bwe.details)
        except Exception as _e:
            traceback.print_exc()
            print(_e)

    def get_nightly_links(self):
        response = requests.get(self.night_url, auth=(secrets['ztf_depo']['user'], secrets['ztf_depo']['pwd']))
        html = response.text

        link_list = []
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.findAll('a')

        for link in links:
            txt = link.getText()
            print(txt)
            if len(txt) == 20 and txt not in self.processed_links:
                link_list.append(txt)
            if 'CalQA' in txt:
                link_list.append(txt)
            if 'CovMaps' in txt:
                link_list.append(txt)

        return link_list

    def get_qa_files(self, url):

        try:
            response = requests.get(url, auth=(secrets['ztf_depo']['user'], secrets['ztf_depo']['pwd']))
            html = response.text

            # link_list = []
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.findAll('a')

            for link in links:
                txt = link.getText()

                # get compressed streaks
                if ('strkcutouts' in txt) and (txt not in self.processed_stamps):

                    # link_list.append(txt)

                    if self.transfer_to_local:
                        # fetch
                        print(f'Downloading {txt}')
                        self.logger.debug(f'Downloading {txt}')
                        alerts_link = os.path.join(url, txt)

                        response = requests.get(alerts_link, auth=(secrets['ztf_depo']['user'],
                                                                   secrets['ztf_depo']['pwd']))
                        # try saving:
                        try:
                            with open(os.path.join(self.stamps_dir, txt), 'wb') as f:
                                f.write(response.content)
                        except Exception as _e:
                            print(str(_e))
                            # failed to fetch/save file? try removing the archive --
                            # will try again on next loop iteration
                            try:
                                os.remove(os.path.join(self.stamps_dir, txt))
                            except OSError:
                                pass

                        # untar
                        print('Unpacking {:s}'.format(txt))
                        self.logger.debug(f'Unpacking {txt}')
                        try:
                            with tarfile.open(os.path.join(self.stamps_dir, txt)) as tar:
                                tar.extractall(path=self.stamps_dir)
                            # move files from unpacked dir
                            base_name = txt.split('.tar.gz')[0]
                            unpacked_dir = os.path.join(self.stamps_dir, base_name)
                            files = os.listdir(unpacked_dir)
                            for f in files:
                                shutil.move(os.path.join(unpacked_dir, f), self.stamps_dir)
                            # delete dir
                            shutil.rmtree(unpacked_dir)
                        except Exception as _e:
                            print(str(_e))
                            # failed to unpack? will try again on next loop iteration
                            try:
                                # remove archive:
                                os.remove(os.path.join(self.stamps_dir, txt))
                                # remove whatever got unpacked:
                                shutil.rmtree(os.path.join(self.stamps_dir, txt.split('.')[0]))
                            except OSError:
                                pass

                                # get compressed streaks

                if ('streaks' in txt) and (txt not in self.processed_meta):

                    if self.transfer_to_local:
                        # fetch
                        print(f'Downloading {txt}')
                        self.logger.debug(f'Downloading {txt}')
                        alerts_link = os.path.join(url, txt)
                        response = requests.get(alerts_link, auth=(secrets['ztf_depo']['user'],
                                                                   secrets['ztf_depo']['pwd']))
                        # try saving:
                        try:
                            with open(os.path.join(self.meta_dir, txt), 'wb') as f:
                                f.write(response.content)
                        except Exception as _e:
                            print(str(_e))
                            # failed to fetch/save file? try removing the archive --
                            # will try again on next loop iteration
                            try:
                                os.remove(os.path.join(self.meta_dir, txt))
                            except OSError:
                                pass

        except Exception as e:
            traceback.print_exc()
            print(e)
            self.logger.error(str(e))

    def clean_up(self):
        try:
            # stuff left in the last batch?
            if len(self.documents) > 0:
                print(f'inserting batch')
                self.logger.info(f'inserting batch')
                self.insert_multiple_db_entries(_collection=self.collection_alerts, _db_entries=self.documents)
                self.documents = []

            # creating alert id index:
            self.db['db'][self.collection_alerts].create_index([('objectId', 1)])
            self.db['db'][self.collection_alerts].create_index([('candid', 1)])

            # create 2d index:
            print('Creating 2d index')
            self.db['db'][self.collection_alerts].create_index([('coordinates.radec_geojson', '2dsphere')])
            print('All done')

        except Exception as e:
            traceback.print_exc()
            print(e)
            self.logger.error(str(e))

        finally:
            try:
                # disconnect from db
                self.disconnect_from_db()
                self.shut_down_logger()
            finally:
                pass


def main(time_skip=False, obsdate=None):

    y = qaWatcher(obsdate=obsdate, data_dir='/data/streaks/', batch_size=128)
    s = time.time()
    links = y.get_nightly_links()

    ccds = ['ccd01', 'ccd02', 'ccd03', 'ccd04',
            'ccd05', 'ccd06', 'ccd07', 'ccd08',
            'ccd09', 'ccd10', 'ccd11', 'ccd12',
            'ccd13', 'ccd14', 'ccd15', 'ccd16']

    for l in reversed(links):
        print(l)
        if 'CalQA' in l and int(datetime.datetime.utcnow().strftime('%H')) >= 7:
            print('Skipping Cals')
            continue

        if time_skip:
            if len(l) == 20:
                try:
                    start, end = l.split('-')
                    now = datetime.datetime.utcnow().strftime('%Hh%Mm%Ss')
                    now_datetime = datetime.datetime.strptime(now, '%Hh%Mm%Ss')
                    end_datetime = datetime.datetime.strptime(end, '%Hh%Mm%Ss/')

                    diff = now_datetime - end_datetime
                    if diff.seconds > 3600:
                        print('Skipping %s' % l)
                        continue
                except Exception as e:
                    print(str(e))
                    pass

        if 'Cov' in l:
            pass
            # ccd_link = '%s/%s/' % (y.night_url, l)
            # y.get_coverage_files(ccd_link)
        else:
            for ccd in ccds:
                # ccd_link = '%s/%s/%s' % (y.night_url, l, ccd)
                ccd_link = os.path.join(y.night_url, l, ccd)
                # qa_files = y.get_qa_files(ccd_link)
                y.get_qa_files(ccd_link)

    # print(qa_files)
    print(f'Loop took {time.time() - s} seconds')
    # clean up
    # y.clean_up()
    # sleep
    time.sleep(10)


if __name__ == '__main__':

    parser = ArgumentParser(description='Ingest AVRO packet into DB')
    parser.add_argument('--obsdate', help='observing date')
    parser.add_argument('--enforce', action='store_true', help='enforce execution')

    args = parser.parse_args()
    obs_date = args.obsdate
    # print(obs_date)

    while True:
        if args.enforce or datetime.datetime.utcnow().hour < 20:
            try:
                main(obsdate=obs_date)
            except Exception as e:
                print(str(e))
                time.sleep(60)
        else:
            print('Sleeping till next night')
            time.sleep(60)
