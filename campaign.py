import pymongo
import math
import datetime
import requests
import json
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
import re
from typing import Union


date_type = Union[datetime.datetime, float]


def jd_to_date(jd):
    """
    Convert Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.

    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.

    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)

    """
    jd += 0.5

    F, I = math.modf(jd)
    I = int(I)

    A = math.trunc((I - 1867216.25) / 36524.25)

    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + F - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day


def jd2date(jd):

    year, month, day = jd_to_date(jd)

    return datetime.datetime(year, month, int(np.floor(day)))


def fetch_cutout(_id: str, date: date_type, _path_out: str='./'):
    _base_url = f"{secrets['deep_asteroids_service']['protocol']}://" + \
                f"{secrets['deep_asteroids_service']['host']}:{secrets['deep_asteroids_service']['port']}"
    _base_url = os.path.join(_base_url, 'data/stamps')

    # print(type(date))
    if isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
        date_utc = date
    elif isinstance(date, float):
        date_utc = jd2date(date).strftime('%Y%m%d')

    try:
        url = os.path.join(_base_url, f'stamps_{date_utc}/{_id}_scimref.jpg')

        print(url)

        filename = os.path.join(_path_out, f'{_id}_scimref.jpg')
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)

                return True

    except Exception as e:
        print(str(e))

    return False


def fetch_real_streakids(date_start=datetime.datetime(2018, 5, 31),
                         date_end=datetime.datetime.utcnow(),
                         _path_out='./'):
    try:
        # date_start = datetime.datetime(2018, 5, 31)
        # date_start = datetime.datetime(2018, 11, 1)
        # date_end = datetime.datetime.utcnow()

        session = requests.Session()
        session.auth = (secrets['yupana']['user'], secrets['yupana']['pwd'])

        reals = dict()

        for dd in range((date_end - date_start).days + 1):

            date = (date_start + datetime.timedelta(days=dd)).strftime('%Y%m%d')

            try:
                url = secrets['yupana']['url']
                result = session.get(url, params={'date': date})
                if result.status_code == 200:
                    # print(result.content)
                    soup = BeautifulSoup(result.content, 'html.parser')
                    # cutouts = re.findall(r'stamps_(.*)//(strkid.*)_scimref', str(soup))
                    cutouts = re.findall(r'(strkid.*)_scimref', str(soup))
                    print(date)
                    print(cutouts)
                    if len(cutouts) > 0:
                        reals[date] = cutouts
            except Exception as e:
                print(str(e))

        json_filename = f'reals_{date_start.strftime("%Y%m%d")}_{date_end.strftime("%Y%m%d")}.json'

        with open(os.path.join(_path_out, json_filename), 'w') as outfile:
            json.dump(reals, outfile, sort_keys=True, indent=2)

        real_ids = []
        for date in reals:
            real_ids += reals[date]

        print('\n', real_ids)

        return {'status': 'success', 'path_json': os.path.join(_path_out, json_filename)}

    except Exception as e:
        print(str(e))

    return {'status': 'failed'}


def fetch_reals(path_json, path_out='./'):

    with open(path_json, 'r') as f:
        data = json.load(f)

    dates = sorted(list(data.keys()))

    _path_out = os.path.join(path_out, f'reals_{dates[0]}_{dates[-1]}')
    if not os.path.exists(_path_out):
        os.makedirs(_path_out)

    for date in data:
        if len(data[date]) > 0:
            print(date)
            for streak_id in data[date]:
                try:
                    print(f'fetching {streak_id}')
                    fetch_cutout(streak_id, datetime.datetime.strptime(date, '%Y%m%d'), _path_out)

                except Exception as e:
                    print(str(e))
                    continue


def sample(date_start=datetime.datetime(2018, 5, 31),
           date_end=datetime.datetime.utcnow(),
           n_samples: int=1000,
           path_out='./'):

    try:
        jd_start = Time(date_start, format='datetime', scale='utc').jd
        jd_end = Time(date_end, format='datetime', scale='utc').jd
        print(jd_start, jd_end)

        client = pymongo.MongoClient(host=secrets['deep_asteroids_mongodb']['host'],
                                     port=secrets['deep_asteroids_mongodb']['port'])

        db = client['deep-asteroids']
        db.authenticate(name=secrets['deep_asteroids_mongodb']['user'],
                        password=secrets['deep_asteroids_mongodb']['pwd'])

        ''' training sets for the rb classifiers (real/bogus) '''
        rb_classifiers = ('rb_vgg6', 'rb_resnet50')

        # rb > 0.8: n_samples cutouts
        # high score by either of the classifiers in the family
        high_rb_score = [{rb_classifier: {'$gt': 0.8}} for rb_classifier in rb_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_rb_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'rb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_gt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        # rb < 0.8: n_samples cutouts
        # low score by either of the classifiers in the family
        low_rb_score = [{classifier: {'$lt': 0.8}} for classifier in rb_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': low_rb_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'rb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_lt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        ''' training sets for the sl classifier (short/long) '''
        sl_classifiers = ('sl_vgg6', 'sl_resnet50')

        # rb > 0.9, sl > 0.8: n_samples cutouts
        high_rb_score = [{rb_classifier: {'$gt': 0.9}} for rb_classifier in rb_classifiers]
        high_sl_score = [{sl_classifier: {'$gt': 0.8}} for sl_classifier in sl_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_rb_score},
                                 {'$or': high_sl_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'sl_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') +
                            '__rb_gt_0.9__sl_gt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        # rb > 0.9, sl < 0.8: n_samples cutouts
        low_sl_score = [{sl_classifier: {'$lt': 0.8}} for sl_classifier in sl_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_rb_score},
                                 {'$or': low_sl_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'sl_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') +
                            '__rb_gt_0.8__sl_lt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            # todo: pyprind instead
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        ''' training sets for the kd classifier (keep/ditch) '''
        kd_classifiers = ('kd_vgg6', 'kd_resnet50')

        # rb > 0.9, sl > 0.9, kd > 0.8: n_samples cutouts
        high_rb_score = [{rb_classifier: {'$gt': 0.9}} for rb_classifier in rb_classifiers]
        high_sl_score = [{sl_classifier: {'$gt': 0.9}} for sl_classifier in sl_classifiers]
        high_kd_score = [{kd_classifier: {'$gt': 0.8}} for kd_classifier in kd_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_rb_score},
                                 {'$or': high_sl_score},
                                 {'$or': high_kd_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'kd_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') +
                            '__rb_gt_0.9__sl_gt_0.9__kd_gt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        # rb > 0.9, sl > 0.9, sl < 0.8: n_samples cutouts
        low_kd_score = [{kd_classifier: {'$lt': 0.8}} for kd_classifier in kd_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_rb_score},
                                 {'$or': high_sl_score},
                                 {'$or': low_kd_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'kd_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') +
                            '__rb_gt_0.9__sl_gt_0.9__kd_lt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        for si, streak in enumerate(streaks):
            print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)

        return {'status': 'success'}

    except Exception as e:
        print(str(e))

    return {'status': 'failed'}


def main(date_start=datetime.datetime(2018, 5, 31),
         date_end=datetime.datetime.utcnow()):

    date_now_utc = datetime.datetime.utcnow()
    path_campaign = os.path.join('./data-raw', f'campaign_{date_now_utc.strftime("%Y%m%d_%H%M%S")}')
    if not os.path.exists(path_campaign):
        os.makedirs(path_campaign)

    print('Fetching real streakid\'s')
    r = fetch_real_streakids(date_start=date_start,
                             date_end=date_end,
                             _path_out=path_campaign)

    if r['status'] == 'success':
        print('Fetching real streaks')
        fetch_reals(r['path_json'], path_out=path_campaign)

    print('Sampling classifications')
    sample(date_start=date_start,
           date_end=date_end,
           n_samples=10,
           path_out=path_campaign)

    print('Uploading to Zwickyverse')
    # todo


if __name__ == '__main__':

    ''' load secrets '''
    with open('./secrets.json') as sjson:
        secrets = json.load(sjson)

    main(date_start=datetime.datetime(2018, 11, 29))
