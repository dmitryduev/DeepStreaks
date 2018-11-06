import json
import os
import requests


def fetch_cutout(_id, date_utc, _path='./'):

    url = f'http://private.caltech.edu:8001/data/stamps/stamps_{date_utc}/{_id}_scimref.jpg'

    filename = os.path.join(_path, f'{_id}_scimref.jpg')
    r = requests.get(url, stream=True)

    if r.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(r.content)


if __name__ == '__main__':

    # path_out = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/reals_20181101_20181105'
    path_out = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/reals_20181106'

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # json_filename = 'reals_20181101_20181105.json'
    json_filename = 'reals_20181106.json'

    with open(json_filename, 'r') as f:
        data = json.load(f)

    for date in data:
        if len(data[date]) > 0:
            print(date)
            for streak_id in data[date]:
                print(f'fetching {streak_id}')
                fetch_cutout(streak_id, date, path_out)
