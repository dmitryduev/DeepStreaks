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
import pyprind
from zwickyverse import Private
import glob


date_type = Union[datetime.datetime, float]


''' load secrets '''
with open('./secrets.json') as sjson:
    secrets = json.load(sjson)


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


def fetch_cutout(_id: str, date: date_type, _path_out: str='./', _v=False):
    _base_url = f"{secrets['deep_asteroids_service']['protocol']}://" + \
                f"{secrets['deep_asteroids_service']['host']}:{secrets['deep_asteroids_service']['port']}"
    _base_url = os.path.join(_base_url, 'data/stamps')

    if _v:
        print(type(date))

    if isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
        date_utc = date.strftime('%Y%m%d')
    elif isinstance(date, float):
        date_utc = jd2date(date).strftime('%Y%m%d')

    try:
        url = os.path.join(_base_url, f'stamps_{date_utc}/{_id}_scimref.jpg')

        if _v:
            print(url)

        filename = os.path.join(_path_out, f'{_id}_scimref.jpg')
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)

                return True

    except Exception as e:
        print(str(e))

    return False


def fetch_real_streakids(date_start=datetime.datetime(2018, 5, 31),
                         date_end=datetime.datetime.utcnow(),
                         _path_out='./', _v: bool=True):
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
                    if _v:
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

        if _v:
            print('\n', real_ids)

        return {'status': 'success', 'path_json': os.path.join(_path_out, json_filename)}

    except Exception as e:
        print(str(e))

    return {'status': 'failed'}


def fetch_reals(path_json, path_out='./', _v: bool=True):

    with open(path_json, 'r') as f:
        data = json.load(f)

    dates = sorted(list(data.keys()))

    _path_out = os.path.join(path_out, f'reals_{dates[0]}_{dates[-1]}')
    if not os.path.exists(_path_out):
        os.makedirs(_path_out)

    if _v:
        bar = pyprind.ProgBar(len(data), stream=1, title='Fetching real streaks...', monitor=True)
    for date in data:
        if len(data[date]) > 0:
            if _v:
                bar.update(iterations=1, item_id=date)
                # print(date)
            for streak_id in data[date]:
                try:
                    # print(f'fetching {streak_id}')
                    fetch_cutout(streak_id, datetime.datetime.strptime(date, '%Y%m%d'), _path_out, _v=False)

                except Exception as e:
                    print(str(e))
                    continue


def sample(date_start=datetime.datetime(2018, 5, 31),
           date_end=datetime.datetime.utcnow(),
           n_samples: int=1000,
           path_out='./', _v: bool=True):

    try:
        jd_start = Time(date_start, format='datetime', scale='utc').jd
        jd_end = Time(date_end, format='datetime', scale='utc').jd
        if _v:
            print(jd_start, jd_end)

        client = pymongo.MongoClient(host=secrets['deep_asteroids_mongodb']['host'],
                                     port=secrets['deep_asteroids_mongodb']['port'])

        db = client['deep-asteroids']
        db.authenticate(name=secrets['deep_asteroids_mongodb']['user'],
                        password=secrets['deep_asteroids_mongodb']['pwd'])

        ''' training sets for the rb classifiers (real/bogus) '''
        rb_classifiers = ('rb_vgg6', 'rb_resnet50', 'rb_densenet121')

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
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for rb___rb_gt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

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
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for rb___rb_lt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

        ''' training sets for the sl classifier (short/long) '''
        sl_classifiers = ('sl_vgg6', 'sl_resnet50', 'sl_densenet121')

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
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for sl__rb_gt_0.9__sl_gt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

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
                            '__rb_gt_0.9__sl_lt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for sl__rb_gt_0.9__sl_lt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

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
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for kd__rb_gt_0.9__sl_gt_0.9__kd_gt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

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
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for kd__rb_gt_0.9__sl_gt_0.9__kd_lt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

        ''' training sets for the os classifiers (one-shot real/bogus) '''
        os_classifiers = ('os_vgg6', 'os_resnet50', 'os_densenet121')

        # os > 0.8: n_samples cutouts
        # high score by either of the classifiers in the family
        high_os_score = [{os_classifier: {'$gt': 0.8}} for os_classifier in os_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': high_os_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'os_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__os_gt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for os___os_gt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

        # os < 0.8: n_samples cutouts
        # low score by either of the classifiers in the family
        low_os_score = [{classifier: {'$lt': 0.8}} for classifier in os_classifiers]
        cursor = db['deep-asteroids'].aggregate([
            {'$match': {'$and': [{'$or': low_os_score},
                                 {'jd': {'$gt': jd_start, '$lt': jd_end}}
                                 ]}},
            {'$project': {'_id': 1, 'jd': 1}},
            {'$sample': {'size': n_samples}}
        ], allowDiskUse=True)

        streaks = list(cursor)

        path = os.path.join(path_out, 'os_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '__os_lt_0.8')
        os.makedirs(path)

        num_streaks = len(streaks)
        if _v:
            bar = pyprind.ProgBar(num_streaks, stream=1,
                                  title='Fetching streaks for os___os_lt_0.8', monitor=True)
        for si, streak in enumerate(streaks):
            # print(f'fetching {streak["_id"]}: {si+1}/{num_streaks}')
            fetch_cutout(streak['_id'], streak['jd'], path)
            if _v:
                bar.update(iterations=1)

        return {'status': 'success'}

    except Exception as e:
        print(str(e))

    return {'status': 'failed'}


def upload_to_zwickyverse(_project_ids: dict, _campaign_name: str='', _path_campaign: str='./',
                          protocol='https', host='private.caltech.edu', port=443,
                          _upload_reals: bool = True, _upload_samples: bool=True, _v: bool=True):

    with Private(protocol=protocol, host=host, port=port,
                 username=secrets['zwickyverse']['user'], password=secrets['zwickyverse']['pwd'], verbose=_v) as p:

        # get metadata of all projects
        projects = p.get_project()
        project_ids_zv = [pid['_id'] for pid in projects]

        # upload reals:
        if _upload_reals:
            for cls, project_id in _project_ids.items():
                # make sure project_id exists:
                if project_id in project_ids_zv:
                    # datasets to upload:
                    dataset_names = [os.path.basename(cls_path) for cls_path
                                     in glob.glob(os.path.join(_path_campaign, f'reals_*')) if os.path.isdir(cls_path)]
                    if _v:
                        print(dataset_names)

                    for dataset_name in dataset_names:
                        path = os.path.abspath(os.path.join(_path_campaign, dataset_name))
                        images = glob.glob(os.path.join(path, '*.jpg'))  # image absolute paths
                        # print(images)
                        ds_id = p.add_dataset(project_id=project_id, name=dataset_name,
                                              description=_campaign_name, files=images)
                        if _v:
                            print(f'created dataset in project {project_id}: {ds_id}')

                else:
                    print(f'project_id {project_id} not found on the server')

        # upload samples:
        if _upload_samples:
            for cls, project_id in _project_ids.items():
                # make sure project_id exists:
                if project_id in project_ids_zv:
                    # datasets to upload:
                    dataset_names = [os.path.basename(cls_path) for cls_path
                                     in glob.glob(os.path.join(_path_campaign, f'{cls}_*'))]
                    if _v:
                        print(dataset_names)

                    for dataset_name in dataset_names:
                        path = os.path.abspath(os.path.join(_path_campaign, dataset_name))
                        images = glob.glob(os.path.join(path, '*.jpg'))  # image absolute paths
                        # print(images)
                        ds_id = p.add_dataset(project_id=project_id, name=dataset_name,
                                              description=_campaign_name, files=images)
                        if _v:
                            print(f'created dataset in project {project_id}: {ds_id}')

                else:
                    print(f'project_id {project_id} not found on the server')


def main(date_start=datetime.datetime(2018, 5, 31),
         date_end=datetime.datetime.utcnow()):

    date_now_utc = datetime.datetime.utcnow()
    campaign_name = f'campaign_{date_now_utc.strftime("%Y%m%d_%H%M%S")}'

    path_campaign = os.path.join('./data-raw', campaign_name)
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
           n_samples=1000,
           path_out=path_campaign)

    print('Uploading to Zwickyverse')
    project_ids = {'rb': '5b96af9c0354c9000b0aea36',
                   'sl': '5b99b2c6aec3c500103a14de',
                   'kd': '5be0ae7958830a0018821794',
                   'os_vgg6': '5c05bbdc826480000a95c0bf'}
    upload_to_zwickyverse(_project_ids=project_ids,
                          _campaign_name=campaign_name, _path_campaign=path_campaign,
                          _upload_reals=True, _upload_samples=True)


if __name__ == '__main__':
    # main(date_start=datetime.datetime(2018, 11, 1))
    main(date_start=datetime.datetime(2018, 12, 1))
