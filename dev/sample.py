import pymongo
import math
import datetime
# import aiohttp
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os


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


def fetch_cutout(_id, jd, _path='./'):

    date_utc = jd2date(jd).strftime('%Y%m%d')
    url = f'http://private.caltech.edu:8001/data/stamps/stamps_{date_utc}/{_id}_scimref.jpg'

    filename = os.path.join(_path, f'{_id}_scimref.jpg')
    r = requests.get(url, stream=True)

    if r.status_code == 200:
        # if (not os.path.exists(os.path.join('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/'+
        #                                     '20181105_161419__rb_gt_0.97__sl_gt_0.85', f'{_id}_scimref.jpg'))) and \
        #         (not os.path.exists(os.path.join('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/' +
        #                                          '20181105_164845__rb_gt_0.97__sl_gt_0.85', f'{_id}_scimref.jpg'))):
        # if (not os.path.exists(os.path.join('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/'+
        #                                     '20181102_124622__rb_gt_0.8', f'{_id}_scimref.jpg'))):
        with open(filename, 'wb') as f:
            f.write(r.content)


if __name__ == '__main__':
    ''' load secrets '''
    with open('./secrets.json') as sjson:
        secrets = json.load(sjson)

    client = pymongo.MongoClient(host=secrets['deep_asteroids_mongodb']['host'],
                                 port=secrets['deep_asteroids_mongodb']['port'])

    db = client['deep-asteroids']
    db.authenticate(name=secrets['deep_asteroids_mongodb']['user'],
                    password=secrets['deep_asteroids_mongodb']['pwd'])

    # cursor = db['deep-asteroids'].find({}, {'_id': 1, 'rb': 1, 'sl': 1})

    ''' training sets for the sl classifier (short/long) '''
    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$gt': 0.93}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': 8000}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    path = os.path.join('data-raw', datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_gt_0.93')
    os.makedirs(path)

    num_streaks = len(streaks)
    for si, streak in enumerate(streaks):
        print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
        fetch_cutout(streak['_id'], streak['jd'], path)

    raise Exception('HAENDE HOCH!!')

    ''' training sets for the kd classifier (keep/ditch) '''
    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$gt': 0.97}, 'sl': {'$gt': 0.85}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': 3333}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    path = os.path.join('data-raw', datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_gt_0.97__sl_gt_0.85')
    os.makedirs(path)

    num_streaks = len(streaks)
    for si, streak in enumerate(streaks):
        print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
        fetch_cutout(streak['_id'], streak['jd'], path)

    raise Exception('HAENDE HOCH!!')

    ''' Fetch rb > 0.8 '''

    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$gt': 0.8}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': 3000}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    path = os.path.join('data-raw', datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_gt_0.8')
    os.makedirs(path)

    num_streaks = len(streaks)
    for si, streak in enumerate(streaks):
        print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
        fetch_cutout(streak['_id'], streak['jd'], path)

    ''' Fetch rb < 0.2 '''

    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$lt': 0.2}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': 3000}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    path = os.path.join('data-raw', datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__rb_lt_0.2')
    os.makedirs(path)

    num_streaks = len(streaks)
    for si, streak in enumerate(streaks):
        print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
        fetch_cutout(streak['_id'], streak['jd'], path)

    ''' Fetch 0.2 < rb < 0.8 '''

    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$gt': 0.2, '$lt': 0.8}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': 1000}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    path = os.path.join('data-raw', datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__0.2_gt_rb_lt_0.8')
    os.makedirs(path)

    num_streaks = len(streaks)
    for si, streak in enumerate(streaks):
        print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
        fetch_cutout(streak['_id'], streak['jd'], path)
