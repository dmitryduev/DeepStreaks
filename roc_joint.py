import itertools
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from keras.models import model_from_json

import json
import glob
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt


def load_model_helper(path, model_base_name):
    # return load_model(path)
    with open(os.path.join(path, f'{model_base_name}.architecture.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    m.load_weights(os.path.join(path, f'{model_base_name}.weights.h5'))

    return m


def load_data(path: str = './data', project_ids: dict = {'os': '5c05bbdc826480000a95c0bf'},
              binary: bool = True, grayscale: bool = True,
              resize: tuple = (144, 144), test_size=0.1, verbose: bool = True, random_state: int = 42):

    files = dict()

    for classifier in project_ids.keys():
        if verbose:
            print(f'Loading project {classifier}: {project_ids[classifier]}')

        # file names:
        x = []
        # classifications:
        y = []

        # get json file with os project metadata
        project_meta_json = os.path.join(path, f'{project_ids[classifier]}.json')
        with open(project_meta_json) as f:
            project_meta = json.load(f)

        path_project = os.path.join(path, project_ids[classifier])

        for dataset_id in project_meta['datasets']:
            if verbose:
                print(f'Loading dataset {dataset_id}')

            dataset_json = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*.json')))
            if len(dataset_json) > 0:
                dataset_json = dataset_json[-1]

                with open(dataset_json) as f:
                    classifications = json.load(f)
                # print(classifications)

                path_dataset = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*')))[0]
                # print(path_dataset)

                for k, v in classifications.items():
                    image_path = os.path.join(path_dataset, k)

                    if os.path.exists(image_path):

                        x.append(k)
                        y.append(1)

        # numpy-fy and split to test/train
        # os 2648
        # rb 2531
        # sl 1642
        # kd 2206

        x = np.array(x)
        y = np.array(y)

        x_train, x_test, _, _ = train_test_split(x, y, test_size=test_size, random_state=random_state)

        if classifier == 'os':
            files[classifier] = set(x_test)
        else:
            files[classifier] = set(x_train)

    for classifier in files:
        print(classifier, len(files[classifier]))

    ''' Load test data for os classifiers that has not been used in training for other classifiers '''

    # data:
    x = []
    # classifications:
    y = []

    # get json file with project metadata
    project_meta_json = os.path.join(path, f'{project_ids["os"]}.json')
    with open(project_meta_json) as f:
        project_meta = json.load(f)

    # print(project_meta)

    classes_list = project_meta['classes']

    # if it's a binary problem, do {'class1': 0, 'class2': 1}
    # if multi-class (N), do {'class1': np.array([1, 0, ..., 0]), 'class2': np.array([0, 1, ..., 0]),
    #                         'classN': np.array([0, 0, ..., 1])}
    if binary:
        classes = {classes_list[0]: 0, classes_list[1]: 1}
    else:
        classes = {}
        n_c = len(classes_list)

        for ci, cls in enumerate(classes_list):
            classes[cls] = np.zeros(n_c)
            classes[cls][ci] = 1

    # print(classes)

    path_project = os.path.join(path, project_ids['os'])

    for dataset_id in project_meta['datasets']:
        if verbose:
            print(f'Loading dataset {dataset_id}')

        dataset_json = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*.json')))
        if len(dataset_json) > 0:
            dataset_json = dataset_json[-1]

            with open(dataset_json) as f:
                classifications = json.load(f)
            # print(classifications)

            path_dataset = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*')))[0]
            # print(path_dataset)

            for k, v in classifications.items():
                # resize and normalize:
                image_path = os.path.join(path_dataset, k)

                if os.path.exists(image_path) and (k in files['os']) and \
                        (k not in files['rb']) and (k not in files['sl']) and (k not in files['kd']):

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

                    x.append(img)

                    image_class = classes[v[0]]
                    y.append(image_class)

    # load reals not used in training:
    path_reals = './data-raw/reals_20181201_20190124'
    reals = glob.glob(os.path.join(path_reals, '*.jpg'))

    for image_path in reals:

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

                x.append(img)

                image_class = 1
                y.append(image_class)

            except:
                continue

    # numpify

    x = np.array(x)
    y = np.array(y)

    if verbose:
        print(x.shape)
        print(y.shape)

        # check statistics on different classes
        if not binary:
            print('\n')
            for i, ii in enumerate(sorted(classes.keys())):
                print(f'{ii}:', np.sum(y[:, i]))
            print('\n')
        else:
            print('\n')
            cs = list(classes.keys())
            print(f'{cs[0]}:', len(y) - np.sum(y))
            print(f'{cs[1]}:', np.sum(y))
            print('\n')

    return x, y


if __name__ == '__main__':

    tf.keras.backend.clear_session()

    # load test data that has never been seen by either classifier at training stage

    x_, y_ = load_data(project_ids={'os': '5c05bbdc826480000a95c0bf',
                                    'rb': '5b96af9c0354c9000b0aea36',
                                    'sl': '5b99b2c6aec3c500103a14de',
                                    'kd': '5be0ae7958830a0018821794'})

    path_base = './'

    with open(os.path.join(path_base, 'service/code/config.json')) as f:
        config = json.load(f)

    models = config['models']
    # models = config['models_201901']

    models["os_vgg6"] = "5c05bbdc826480000a95c0bf_VGG6_20190117_232340"
    models["os_resnet50"] = "5c05bbdc826480000a95c0bf_ResNet50_20190118_000629"
    models["os_densenet121"] = "5c05bbdc826480000a95c0bf_densenet121_20190118_043746"

    model_names = list(models.keys())

    path_models = os.path.join(path_base, 'service/models')

    c_families = {'rb': '5b96af9c0354c9000b0aea36',
                  'sl': '5b99b2c6aec3c500103a14de',
                  'kd': '5be0ae7958830a0018821794',
                  'os': '5c05bbdc826480000a95c0bf'}

    # # mpl colors:
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # # [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
    # #  u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    # # line styles:
    # line_styles = ['-', '--', ':']
    #
    # # thresholds
    # score_thresholds = [0.99, 0.9, 0.5, 0.1, 0.01]
    #
    # # ROC
    # fig = plt.figure()
    # fig.subplots_adjust(bottom=0.09, left=0.05, right=0.76, top=0.98, wspace=0.2, hspace=0.2)
    # lw = 1.6
    # # ROCs
    # ax = fig.add_subplot(1, 2, 1)
    # # zoomed ROCs
    # ax2 = fig.add_subplot(1, 2, 2)
    #
    # ax.plot([0, 1], [0, 1], color='#333333', lw=lw, linestyle='--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate (Contamination)')
    # ax.set_ylabel('True Positive Rate (Sensitivity)')
    # # ax.legend(loc="lower right")
    # # ax.legend(loc="best")
    # ax.grid(True)
    #
    # ax2.set_xlim([0.0, .2])
    # ax2.set_ylim([0.8, 1.0])
    # ax2.set_xlabel('False Positive Rate (Contamination)')
    # ax2.set_ylabel('True Positive Rate (Sensitivity)')
    # # ax.legend(loc="lower right")
    # # ax2.legend(loc="best")
    # ax2.grid(True)

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

            y_pred = m.predict(x_, batch_size=32, verbose=True)

            predictions[c_family][model_name] = y_pred

    y_deep_streaks_rb_sl_kd = None
    y_deep_streaks_os = None

    thresholds = {'rb': 0.5, 'sl': 0.5, 'kd': 0.5, 'os': 0.5}

    for fam in ('rb', 'sl', 'kd'):
        yy = None
        for model_name in predictions[fam]:
            yyy = predictions[fam][model_name] > thresholds[fam]
            yy = np.logical_or(yy, yyy) if yy is not None else yyy

        y_deep_streaks_rb_sl_kd = np.logical_and(y_deep_streaks_rb_sl_kd, yy) if y_deep_streaks_rb_sl_kd is not None \
            else yy

    for model_name in predictions['os']:
        yyy = predictions['os'][model_name] > thresholds['os']
        y_deep_streaks_os = np.logical_or(y_deep_streaks_os, yyy) if y_deep_streaks_os is not None else yyy

    # print(y_deep_streaks_rb_sl_kd)
    # print(y_deep_streaks_os)

    # Plot confusion matrices
    fig2 = plt.figure()
    fig2.subplots_adjust(bottom=0.06, left=0.01, right=1.0, top=0.93, wspace=0.0, hspace=0.12)

    confusion_matr = confusion_matrix(y_, y_deep_streaks_rb_sl_kd)
    confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

    ax = fig2.add_subplot(2, 2, 1)
    ax2 = fig2.add_subplot(2, 2, 2)
    ax.imshow(confusion_matr, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.imshow(confusion_matr_normalized, interpolation='nearest', cmap=plt.cm.Blues)

    ax.axis('off')
    ax2.axis('off')

    thresh = confusion_matr.max() / 2.
    thresh_norm = confusion_matr_normalized.max() / 2.
    for i, j in itertools.product(range(confusion_matr.shape[0]), range(confusion_matr.shape[1])):
        ax.text(j, i, format(confusion_matr[i, j], 'd'),
                horizontalalignment="center",
                color="white" if confusion_matr[i, j] > thresh else "black")
        ax2.text(j, i, format(confusion_matr_normalized[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matr_normalized[i, j] > thresh_norm else "black")

    print('Confusion matrix for y_deep_streaks_rb_sl_kd:')
    print(confusion_matr)

    print('Normalized confusion matrix for y_deep_streaks_rb_sl_kd:')
    print(confusion_matr_normalized)

    confusion_matr = confusion_matrix(y_, y_deep_streaks_os)
    confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

    ax3 = fig2.add_subplot(2, 2, 3)
    ax4 = fig2.add_subplot(2, 2, 4)
    ax3.imshow(confusion_matr, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.imshow(confusion_matr_normalized, interpolation='nearest', cmap=plt.cm.Blues)

    ax3.axis('off')
    ax4.axis('off')

    thresh = confusion_matr.max() / 2.
    thresh_norm = confusion_matr_normalized.max() / 2.
    for i, j in itertools.product(range(confusion_matr.shape[0]), range(confusion_matr.shape[1])):
        ax3.text(j, i, format(confusion_matr[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matr[i, j] > thresh else "black")
        ax4.text(j, i, format(confusion_matr_normalized[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matr_normalized[i, j] > thresh_norm else "black")

    print('Confusion matrix for y_deep_streaks_os:')
    print(confusion_matr)

    print('Normalized confusion matrix for y_deep_streaks_os:')
    print(confusion_matr_normalized)

    fig2.savefig(f'./cm_rb_sl_kd__vs__os.png', dpi=300)
    plt.show()
