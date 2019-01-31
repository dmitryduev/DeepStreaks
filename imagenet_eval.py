import os
import glob
import argparse
import requests
import numpy as np
from copy import deepcopy
import json
from PIL import Image, ImageOps

import tensorflow as tf
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix


def load_model_helper(path, model_base_name):
    # return load_model(path)
    with open(os.path.join(path, f'{model_base_name}.architecture.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    m.load_weights(os.path.join(path, f'{model_base_name}.weights.h5'))

    return m


def thres(v, thr: float = 0.5):
    v_ = np.array(deepcopy(v))

    v_[v_ >= thr] = 1
    v_[v_ < thr] = 0

    return v_


def load_imagenet_data(path: str = './paper/imagenet', resize: tuple = (144, 144), grayscale: bool = True):
    cats = [os.path.basename(p).split('.txt')[0] for p in glob.glob(os.path.join(path, '*.txt'))]

    data_files = {cat: sorted(glob.glob(os.path.join(path, cat, '*.jpg'))) for cat in cats}

    # print(data_files)

    data = dict()

    for cat in data_files:

        data[cat] = {'x': [], 'y': []}

        for image_path in data_files[cat]:
            # resize and normalize:

            if os.path.exists(image_path):

                try:

                    if grayscale:
                        # the assumption is that images are grayscale
                        img = np.array(ImageOps.grayscale(Image.open(image_path)).resize(resize, Image.BILINEAR)) / 255.
                        img = np.expand_dims(img, 2)

                    else:
                        # make it rgb:
                        img = ImageOps.grayscale(Image.open(image_path)).resize(resize, Image.BILINEAR)
                        rgbimg = Image.new("RGB", img.size)
                        rgbimg.paste(img)
                        img = np.array(rgbimg) / 255.

                    data[cat]['x'].append(img)

                except Exception as e:
                    print(str(e))
                    continue

        data[cat]['x'] = np.array(data[cat]['x'])
        # data[cat]['y'] = np.zeros_like(data[cat]['x'])

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepStreaks: eval on imagenet')

    parser.add_argument('--fetch', action='store_true', help='Fetch')
    parser.add_argument('--eval', action='store_true', help='Eval')

    args = parser.parse_args()

    if args.fetch:
        path_imagenet = './paper/imagenet'

        category_urls = glob.glob(os.path.join(path_imagenet, '*.txt'))

        for category_url in category_urls:
            category = os.path.basename(category_url).split('.txt')[0]
            print(category)

            path_category = os.path.join(path_imagenet, category)
            if not os.path.exists(path_category):
                os.mkdir(path_category)

            with open(category_url, 'r') as f:
                urls = f.read()
            urls = urls.split('\n')

            # print(urls[:2])

            ni = 1
            for url in urls:
                try:
                    if url.endswith('.jpg'):
                        r = requests.get(url, timeout=2)
                        if r.status_code == 200:
                            print(f'downloading image #{ni:04d} in {category} category')
                            with open(os.path.join(path_category, f'{ni:04d}.jpg'), 'wb') as f:
                                f.write(r.content)
                            ni += 1
                except Exception as e:
                    print(str(e))

    if args.eval:

        tf.keras.backend.clear_session()

        # path_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/'
        path_base = './'
        path_models = os.path.join(path_base, 'service/models')

        with open(os.path.join(path_base, 'service/code/config.json')) as f:
            config = json.load(f)

        models = config['models']
        # models = config['models_201901']
        model_names = list(models.keys())

        c_families = {'rb': '5b96af9c0354c9000b0aea36',
                      'sl': '5b99b2c6aec3c500103a14de',
                      'kd': '5be0ae7958830a0018821794',
                      'os': '5c05bbdc826480000a95c0bf'}

        # load data
        data = load_imagenet_data()

        predictions = dict()

        for cfi, c_family in enumerate(c_families):

            project_id = c_families[c_family]

            print(c_family, project_id)

            mn = [m_ for m_ in model_names if c_family in m_]
            n_mn = len(mn)

            predictions[c_family] = dict()

            for ii, model_name in enumerate(mn):
                print(f'loading model {model_name}: {models[model_name]}')
                m = load_model_helper(path_models, models[model_name])

                predictions[c_family][model_name] = dict()

                for cat in data:
                    x_test = data[cat]['x']

                    y = m.predict(x_test, batch_size=32, verbose=True)

                    # for thr in (0.5, 0.9):
                    # for thr in (0.5,):
                    thr = 0.5
                    labels_pred = thres(y, thr=thr)
                    confusion_matr = confusion_matrix(np.zeros_like(labels_pred), labels_pred)
                    confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:,
                                                                                 np.newaxis]

                    print(f'Threshold: {thr}')
                    print('Confusion matrix:')
                    print(confusion_matr)

                    print('Normalized confusion matrix:')
                    print(confusion_matr_normalized)

                    predictions[c_family][model_name][cat] = y

        thresholds = {'rb': 0.5, 'sl': 0.5, 'kd': 0.5, 'os': 0.5}

        for cat in data:
            print(f'\n\nCategory: {cat}')

            y_deep_streaks_rb_sl_kd = None
            y_deep_streaks_os = None

            for fam in ('rb', 'sl', 'kd'):
                yy = None
                for model_name in predictions[fam]:
                    yyy = predictions[fam][model_name][cat] > thresholds[fam]
                    yy = np.logical_or(yy, yyy) if yy is not None else yyy

                y_deep_streaks_rb_sl_kd = np.logical_and(y_deep_streaks_rb_sl_kd, yy) \
                    if y_deep_streaks_rb_sl_kd is not None \
                    else yy

            confusion_matr = confusion_matrix(np.zeros_like(y_deep_streaks_rb_sl_kd), y_deep_streaks_rb_sl_kd)
            confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

            print('Confusion matrix for y_deep_streaks_rb_sl_kd:')
            print(confusion_matr)

            print('Normalized confusion matrix for y_deep_streaks_rb_sl_kd:')
            print(confusion_matr_normalized)

            # for model_name in predictions['os']:
            #     yyy = predictions['os'][model_name][cat] > thresholds['os']
            #     y_deep_streaks_os = np.logical_or(y_deep_streaks_os, yyy) if y_deep_streaks_os is not None else yyy
            #
            # confusion_matr = confusion_matrix(np.zeros_like(y_deep_streaks_os), y_deep_streaks_os)
            # confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]
            #
            # print('Confusion matrix for y_deep_streaks_os:')
            # print(confusion_matr)
            #
            # print('Normalized confusion matrix for y_deep_streaks_os:')
            # print(confusion_matr_normalized)

            y_deep_streaks_os = None
            for model_name in predictions['os']:
                # print(model_name)
                if 'vgg6' in model_name:
                    continue
                yyy = predictions['os'][model_name][cat] > thresholds['os']
                y_deep_streaks_os = np.logical_or(y_deep_streaks_os, yyy) if y_deep_streaks_os is not None else yyy

            confusion_matr = confusion_matrix(np.zeros_like(y_deep_streaks_os), y_deep_streaks_os)
            confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

            print('Confusion matrix for y_deep_streaks_os:')
            print(confusion_matr)

            print('Normalized confusion matrix for y_deep_streaks_os:')
            print(confusion_matr_normalized)
