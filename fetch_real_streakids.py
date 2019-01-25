from bs4 import BeautifulSoup
import requests
import re
import datetime
import json
import os


if __name__ == '__main__':

    ''' load secrets '''
    with open('./secrets.json') as sjson:
        secrets = json.load(sjson)

    # date_start = datetime.datetime(2018, 1, 31)
    # date_start = datetime.datetime(2018, 5, 31)
    # date_start = datetime.datetime(2018, 11, 1)
    date_start = datetime.datetime(2018, 12, 1)
    # date_start = datetime.datetime(2018, 12, 15)
    # date_end = datetime.datetime(2018, 4, 30)
    date_end = datetime.datetime.utcnow()

    session = requests.Session()
    session.auth = (secrets['yupana']['user'], secrets['yupana']['pwd'])

    reals = dict()

    for dd in range((date_end - date_start).days + 1):

        # date = '20180929'
        date = (date_start + datetime.timedelta(days=dd)).strftime('%Y%m%d')

        try:
            # url = 'http://private.caltech.edu:8088/zstreak/shepherd.cgi'
            url = 'http://yupana.caltech.edu/cgi-bin/ptf/ssm/zsrs/shepherd.cgi'
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

    with open(json_filename, 'w') as outfile:
        json.dump(reals, outfile, sort_keys=True, indent=2)

    real_ids = []
    for date in reals:
        real_ids += reals[date]

    print('\n', real_ids)
