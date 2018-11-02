import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageOps, Image
import numpy as np
import pymongo
import datetime
import math
import os
import json
import glob
import requests
from io import BytesIO


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


# path_im = '/Users/dmitryduev/_caltech/python/deep-asteroids/strkid6361363656150004_pid636136365615_scimref.jpg'
# # path_im = '/Users/dmitryduev/_caltech/python/deep-asteroids/strkid6354544708150001_pid635454470815_scimref.jpg'
# image = mpimg.imread(path_im)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(image)
#
# plt.show()


if __name__ == '__main__':
    ''' load secrets '''
    with open('./secrets.json') as sjson:
        secrets = json.load(sjson)

    path_out = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/synthetic_streaks_hard_cases_20181101'

    client = pymongo.MongoClient(host=secrets['deep_asteroids_mongodb']['host'],
                                 port=secrets['deep_asteroids_mongodb']['port'])

    db = client['deep-asteroids']
    db.authenticate(name=secrets['deep_asteroids_mongodb']['user'],
                    password=secrets['deep_asteroids_mongodb']['pwd'])

    # cursor = db['deep-asteroids'].find({}, {'_id': 1, 'rb': 1, 'sl': 1})

    ''' Fetch rb > 0.99 and rb < 0.001 '''
    n_streaks = 1000

    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$gt': 0.97}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': n_streaks}}
    ], allowDiskUse=True)

    streaks = list(cursor)

    cursor = db['deep-asteroids'].aggregate([
        {'$match': {'rb': {'$lt': 0.001}}},
        {'$project': {'_id': 1, 'jd': 1}},
        {'$sample': {'size': n_streaks}}
    ], allowDiskUse=True)

    boguses = list(cursor)

    ''' magic '''
    for si, streak in enumerate(streaks):
        print(f'Case: {si}')
        date_utc = jd2date(streak['jd']).strftime('%Y%m%d')
        url = f'http://private.caltech.edu:8001/data/stamps/stamps_{date_utc}/{streak["_id"]}_scimref.jpg'

        filename = os.path.join(path_out, f'{len(glob.glob(os.path.join(path_out, "*.jpg"))) + 1}.jpg')
        print(filename)

        r = requests.get(url, stream=True)

        if r.status_code == 200:
            # with open(filename, 'wb') as f:
            #     f.write(r.content)

            image_bytes = BytesIO(r.content)

            # image = mpimg.imread(image_bytes)

            img = Image.open(image_bytes)
            x = np.array(ImageOps.grayscale(img))
            x_flat = x.flatten()
            # print(x.shape)

            bogus = boguses[si]

            date_utc_b = jd2date(bogus['jd']).strftime('%Y%m%d')
            url_b = f'http://private.caltech.edu:8001/data/stamps/stamps_{date_utc_b}/{bogus["_id"]}_scimref.jpg'

            r_b = requests.get(url_b, stream=True)

            if r_b.status_code == 200:
                image_bytes_bogus = BytesIO(r_b.content)

                # image = mpimg.imread(image_bytes)

                img_bogus = Image.open(image_bytes_bogus)
                x_bogus = np.array(ImageOps.grayscale(img_bogus))
                x_bogus_flat = x_bogus.flatten()
                # print(x_bogus.shape)

                if x.shape == x_bogus.shape:

                    fig = plt.figure(figsize=(14, 5))

                    ax1 = plt.subplot2grid((2, 6), (0, 0))
                    ax1.imshow(img)

                    n, bins = np.histogram(x_flat, bins=256)
                    x_m = np.array([xx for xx in x_flat if 0 < xx < 255])
                    max_pixel_value = bins[np.argmax(n)]
                    max_pixel_num = np.max(n)
                    # print(n)
                    x_no_max = np.array([nn for nn in n if abs(max_pixel_num - nn) > 1e-3])
                    median_pixel_value = np.median(x_m)
                    print(np.max(n), np.max(x_no_max))
                    print(np.ceil(max_pixel_value), median_pixel_value)

                    ax2 = plt.subplot2grid((2, 6), (1, 0))
                    ax2.imshow(img_bogus)

                    n_bogus, bins_bogus = np.histogram(x_bogus_flat, bins=256)
                    x_m_bogus = np.array([xx for xx in x_bogus_flat if 0 < xx < 255])
                    max_pixel_value_bogus = bins_bogus[np.argmax(n_bogus)]
                    max_pixel_num_bogus = np.max(n_bogus)
                    # print(n)
                    x_no_max_bogus = np.array([nn for nn in n_bogus if abs(max_pixel_num_bogus - nn) > 1e-3])
                    median_pixel_value_bogus = np.median(x_m_bogus)
                    print(np.max(n_bogus), np.max(x_no_max_bogus))
                    print(np.ceil(max_pixel_value_bogus), median_pixel_value_bogus)

                    ''' combine! '''
                    # img_sum = x + x_bogus

                    # fill in masked stars:
                    max_streak = np.ceil(max_pixel_value)
                    max_bogus = np.ceil(max_pixel_value_bogus)
                    max_masked = np.max((max_streak, max_bogus))
                    print(max_streak, max_bogus, max_masked)

                    # from streak image:
                    if np.max(n) > np.max(x_no_max) * 1.2:
                        x[x == max_streak] = max_masked
                        x[x == max_streak - 1] = max_masked
                        x[x == max_streak + 1] = max_masked
                        # print('land ho!')
                    # from bogus image:
                    if np.max(n_bogus) > np.max(x_no_max_bogus) * 1.2:
                        x_bogus[x_bogus == max_bogus] = max_masked
                        x_bogus[x_bogus == max_bogus - 1] = max_masked
                        x_bogus[x_bogus == max_bogus + 1] = max_masked
                        # print('yo!')

                    # img_sum = x + x_bogus
                    img_sum = np.minimum(x, x_bogus)
                    img_sum[x == max_masked] = max_masked
                    img_sum[x_bogus == max_masked] = max_masked
                    # img_sum[x == max_masked + 1] = max_masked
                    # img_sum[x_bogus == max_masked + 1] = max_masked
                    # img_sum[x == max_masked - 1] = max_masked
                    # img_sum[x_bogus == max_masked - 1] = max_masked

                    # ax3 = plt.subplot2grid((2, 6), (0, 1), colspan=2, rowspan=2)
                    # ax3.imshow(img_sum)

                    # make hist a bit more real:
                    median_pixel_value_sum = np.median(img_sum.flatten()[1:-1])
                    img_sum[img_sum == max_masked] = median_pixel_value_sum

                    ax3 = plt.subplot2grid((2, 6), (0, 1), colspan=2, rowspan=2)
                    ax3.imshow(img_sum)

                    ax4 = plt.subplot2grid((2, 6), (0, 3), colspan=3, rowspan=2)
                    n_sum, bins_sum = np.histogram(img_sum, bins=256)
                    # print(n_sum.shape, bins_sum.shape)
                    ax4.bar(bins_sum[:-1], n_sum)

                    plt.show()

                    I8 = (((img_sum - img_sum.min()) / (img_sum.max() - img_sum.min())) * 255.9).astype(np.uint8)

                    img_out = Image.fromarray(I8)
                    img_out.save(filename)
