from argparse import ArgumentParser
import aiohttp
import asyncio
import time
import os
import shutil
import glob
import datetime
import json
import tarfile
from bs4 import BeautifulSoup
import traceback


''' load config and secrets '''
with open('/app/config.json') as cjson:
    config = json.load(cjson)

# with open('/Users/dmitryduev/_caltech/python/deep-asteroids/service/secrets.json') as sjson:
with open('/app/secrets.json') as sjson:
    secrets = json.load(sjson)


def time_stamps():
    """
    :return: local time, UTC time
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S'), \
           datetime.datetime.utcnow().strftime('%Y%m%d_%H:%M:%S')


async def main(obsdate=None, looponce=False, data_dir='/data/streaks/'):
    base_url = secrets['ztf_depo']['url']

    while True:
        try:
            if obsdate is not None:
                obsdate = obsdate
            else:
                obsdate = datetime.datetime.utcnow().strftime('%Y%m%d')

            stamps_dir = os.path.join(data_dir, 'stamps', f'stamps_{obsdate}')
            meta_dir = os.path.join(data_dir, 'meta', obsdate)

            # Check that the directory exists
            if not os.path.exists(stamps_dir):
                os.makedirs(stamps_dir)
            if not os.path.exists(meta_dir):
                os.makedirs(meta_dir)

            night_url = os.path.join(base_url, obsdate)

            time_range_links = []

            processed_stamps = [os.path.basename(x) for x in glob.glob(os.path.join(stamps_dir, '*strkcutouts*gz'))]
            processed_meta = [os.path.basename(x) for x in glob.glob(os.path.join(meta_dir, '*streaks*'))]

            auth = aiohttp.BasicAuth(login=secrets['ztf_depo']['user'], password=secrets['ztf_depo']['pwd'])

            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.get(night_url) as resp:
                    text = await resp.read()
                    soup = BeautifulSoup(text.decode('utf-8'), 'html5lib')
                    links = soup.findAll('a')
                    # print(links)

                    # get time range links:
                    for link in links:
                        txt = link.getText()
                        if len(txt) == 20 and (txt not in time_range_links):
                            print(txt)
                            time_range_links.append(txt)

                    # iterate over individual time ranges:
                    for time_range_link in time_range_links:
                        print(f'Checking {time_range_link} of {obsdate}')
                        for ccd in range(16):
                            time_range_ccd_link = os.path.join(night_url, time_range_link, f'ccd{ccd+1:02d}')
                            # print(time_range_ccd_link)

                            async with session.get(time_range_ccd_link) as resp_ccd:
                                text_ccd = await resp_ccd.read()
                                soup_ccd = BeautifulSoup(text_ccd.decode('utf-8'), 'html5lib')
                                links_ccd = soup_ccd.findAll('a')

                                for link in links_ccd:
                                    txt = link.getText()

                                    # get compressed streak cutouts
                                    if ('strkcutouts' in txt) and (txt not in processed_stamps):

                                        # fetch
                                        print(*time_stamps(), f'Downloading {txt}')
                                        strkcutout_link = os.path.join(time_range_ccd_link, txt)

                                        # try saving:
                                        try:
                                            async with session.get(strkcutout_link) as resp2:
                                                tmp = await resp2.read()
                                                with open(os.path.join(stamps_dir, txt), 'wb') as f:
                                                    f.write(tmp)
                                        except Exception as _e:
                                            print(str(_e))
                                            # failed to fetch/save file? try removing the archive --
                                            # will try again on next loop iteration
                                            try:
                                                os.remove(os.path.join(stamps_dir, txt))
                                            except OSError:
                                                pass

                                        # untar
                                        print(*time_stamps(), 'Unpacking {:s}'.format(txt))
                                        # self.logger.debug(f'Unpacking {txt}')
                                        try:
                                            with tarfile.open(os.path.join(stamps_dir, txt)) as tar:
                                                tar.extractall(path=stamps_dir)
                                            # move files from unpacked dir
                                            base_name = txt.split('.tar.gz')[0]
                                            unpacked_dir = os.path.join(stamps_dir, base_name)
                                            files = os.listdir(unpacked_dir)
                                            for f in files:
                                                shutil.move(os.path.join(unpacked_dir, f), stamps_dir)
                                            # delete dir
                                            shutil.rmtree(unpacked_dir)
                                        except Exception as _e:
                                            print(str(_e))
                                            # failed to unpack? will try again on next loop iteration
                                            try:
                                                # remove archive:
                                                os.remove(os.path.join(stamps_dir, txt))
                                                # remove whatever got unpacked:
                                                shutil.rmtree(os.path.join(stamps_dir, txt.split('.')[0]))
                                            except OSError:
                                                pass

                                    # get streak meta

                                    if ('streaks' in txt) and (txt not in processed_meta):

                                        # fetch
                                        print(*time_stamps(), f'Downloading {txt}')
                                        meta_link = os.path.join(time_range_ccd_link, txt)

                                        # try saving:
                                        try:
                                            async with session.get(meta_link) as resp2:
                                                tmp = await resp2.read()
                                                with open(os.path.join(meta_dir, txt), 'wb') as f:
                                                    f.write(tmp)
                                        except Exception as _e:
                                            print(str(_e))
                                            # failed to fetch/save file? try removing the archive --
                                            # will try again on next loop iteration
                                            try:
                                                os.remove(os.path.join(meta_dir, txt))
                                            except OSError:
                                                pass

            if looponce:
                break
            else:
                # take a blocking nap
                time.sleep(10)

        except Exception as e:
            traceback.print_exc()
            print(*time_stamps(), str(e))
            # take a blocking nap
            time.sleep(10)


if __name__ == '__main__':
    parser = ArgumentParser(description='Fetch ZTF streak data from IPAC depo')
    parser.add_argument('--obsdate', help='observing date')
    parser.add_argument('--looponce', action='store_true', help='loop once and exit')

    args = parser.parse_args()
    obs_date = args.obsdate
    loop_once = args.looponce

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(obsdate=obs_date, looponce=loop_once))
