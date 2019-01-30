import os
import glob
import argparse
import requests
import numpy as np
from copy import deepcopy
import json
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import model_from_json
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

        data[cat]['x'] = np.array(data[cat]['x'])
        data[cat]['y'] = np.zeros_like(data[cat]['x'])

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

        # load data
        data = load_imagenet_data()

        for ii, model_name in enumerate(model_names):

            print(f'loading model {model_name}: {models[model_name]}')
            m = load_model_helper(path_models, models[model_name])

            for cat in data:
                x_test = data[cat]['x']
                y_test = data[cat]['y']

                y = m.predict(x_test, batch_size=32, verbose=True)

                # for thr in (0.5, 0.9):
                # for thr in (0.5,):
                thr = 0.5
                labels_pred = thres(y, thr=thr)
                confusion_matr = confusion_matrix(y_test, labels_pred)
                confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

                print(f'Threshold: {thr}')
                print('Confusion matrix:')
                print(confusion_matr)

                print('Normalized confusion matrix:')
                print(confusion_matr_normalized)
