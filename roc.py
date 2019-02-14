import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras.models import model_from_json

import json
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import pandas as pd
from copy import deepcopy
import itertools

from utils import load_data

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    tf.keras.backend.clear_session()

    # path_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/'
    path_base = './'

    with open(os.path.join(path_base, 'service/code/config.json')) as f:
        config = json.load(f)

    # models = config['models']
    models = config['models_201901']
    model_names = list(models.keys())

    path_models = os.path.join(path_base, 'service/models')

    c_families = {'rb': '5b96af9c0354c9000b0aea36',
                  'sl': '5b99b2c6aec3c500103a14de',
                  'kd': '5be0ae7958830a0018821794',
                  'os': '5c05bbdc826480000a95c0bf'}
    # c_families = {'rb': '5b96af9c0354c9000b0aea36',
    #               'sl': '5b99b2c6aec3c500103a14de',
    #               'kd': '5be0ae7958830a0018821794'}
    # c_families = {'rb': '5b96af9c0354c9000b0aea36'}

    path_data = './data'

    # mpl colors:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
    #  u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    # line styles:
    line_styles = ['-', '--', ':']

    # thresholds
    score_thresholds = [0.99, 0.9, 0.5, 0.1, 0.01]

    # ROC
    fig = plt.figure(figsize=(14, 5))
    fig.subplots_adjust(bottom=0.09, left=0.05, right=0.70, top=0.98, wspace=0.2, hspace=0.2)
    lw = 1.6
    # ROCs
    ax = fig.add_subplot(1, 2, 1)
    # zoomed ROCs
    ax2 = fig.add_subplot(1, 2, 2)

    ax.plot([0, 1], [0, 1], color='#333333', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Contamination)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    # ax.legend(loc="lower right")
    # ax.legend(loc="best")
    ax.grid(True)

    ax2.set_xlim([0.0, .2])
    ax2.set_ylim([0.8, 1.0])
    ax2.set_xlabel('False Positive Rate (Contamination)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    # ax.legend(loc="lower right")
    # ax2.legend(loc="best")
    ax2.grid(True)

    # Confusion matrices
    fig2 = plt.figure()
    fig2.subplots_adjust(bottom=0.06, left=0.01, right=1.0, top=0.93, wspace=0.0, hspace=0.12)

    cn = 0

    for cfi, c_family in enumerate(c_families):

        project_id = c_families[c_family]

        print(c_family, project_id)

        # load data
        x_train, y_train, x_test, y_test, classes = load_data(path=path_data,
                                                              project_id=project_id,
                                                              binary=True,
                                                              grayscale=True,
                                                              resize=(144, 144),
                                                              test_size=0.1,
                                                              verbose=True,
                                                              random_state=42)

        mn = [m_ for m_ in model_names if c_family in m_]
        n_mn = len(mn)

        for ii, model_name in enumerate(mn):

            print(f'loading model {model_name}: {models[model_name]}')
            m = load_model_helper(path_models, models[model_name])

            y = m.predict(x_test, batch_size=32, verbose=True)

            # for thr in (0.5, 0.9):
            for thr in (0.5,):
                labels_pred = thres(y, thr=thr)
                confusion_matr = confusion_matrix(y_test, labels_pred)
                confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

                print(f'Threshold: {thr}')
                print('Confusion matrix:')
                print(confusion_matr)

                print('Normalized confusion matrix:')
                print(confusion_matr_normalized)

            fpr, tpr, thresholds = roc_curve(y_test, y)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, line_styles[ii], color=colors[cfi], lw=lw)
            ax2.plot(fpr, tpr, line_styles[ii], color=colors[cfi], lw=lw,
                     label=f'{model_name} curve (area = {roc_auc:.5f})')

            # plot thresholds
            for it, thr in enumerate(score_thresholds):
                x_ = np.interp(thr, thresholds[::-1], fpr)
                y_ = np.interp(thr, thresholds[::-1], tpr)
                # print(thr, x_, y_)
                if cfi == 0 and ii == 0:
                    ax.plot(x_, y_, '.', markersize=8, color=colors[-(it + 1)], label=f'Threshold: {1-thr:.2f}')
                else:
                    ax.plot(x_, y_, '.', markersize=8, color=colors[-(it + 1)])

                ax2.plot(x_, y_, 'o', markersize=8, color=colors[-(it + 1)])

            # plot confusion matrices
            ax_ = fig2.add_subplot(3, 2 * len(c_families), ii * 8 + cfi * 2 + 1)
            ax2_ = fig2.add_subplot(3, 2 * len(c_families), ii * 8 + cfi * 2 + 2)

            ax_.imshow(confusion_matr, interpolation='nearest', cmap=plt.cm.Blues)
            ax2_.imshow(confusion_matr_normalized, interpolation='nearest', cmap=plt.cm.Blues)

            tick_marks = np.arange(2)
            # ax_.set_xticks(tick_marks, tick_marks)
            # ax_.set_yticks(tick_marks, tick_marks)
            # ax2_.set_xticks(tick_marks, tick_marks)
            # ax2_.set_yticks(tick_marks, tick_marks)
            #
            # ax_.xaxis.set_visible(False)
            # ax_.yaxis.set_visible(False)
            # ax2_.xaxis.set_visible(False)
            # ax2_.yaxis.set_visible(False)

            ax_.axis('off')
            ax2_.axis('off')

            thresh = confusion_matr.max() / 2.
            thresh_norm = confusion_matr_normalized.max() / 2.
            for i, j in itertools.product(range(confusion_matr.shape[0]), range(confusion_matr.shape[1])):
                ax_.text(j, i, format(confusion_matr[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if confusion_matr[i, j] > thresh else "black")
                ax2_.text(j, i, format(confusion_matr_normalized[i, j], '.2f'),
                          horizontalalignment="center",
                          color="white" if confusion_matr_normalized[i, j] > thresh_norm else "black")

            # if ii == 0:
            #     break

    ax.legend(loc='lower right')
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.savefig(f'./roc_rb_sl_kd.png', dpi=300)
    fig2.savefig(f'./cm_rb_sl_kd.png', dpi=300)

    plt.show()
