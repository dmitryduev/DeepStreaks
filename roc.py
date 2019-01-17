import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
from copy import deepcopy

from utils import load_data


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

    # tf.keras.backend.clear_session()

    path_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/'

    with open(os.path.join(path_base, 'service/code/config.json')) as f:
        config = json.load(f)

    models = config['models']
    model_names = list(models.keys())

    path_models = os.path.join(path_base, 'service/models')

    # c_families = {'rb': '5b96af9c0354c9000b0aea36',
    #               'sl': '5b99b2c6aec3c500103a14de',
    #               'kd': '5be0ae7958830a0018821794',
    #               'os': '5c05bbdc826480000a95c0bf'}
    c_families = {'rb': '5b96af9c0354c9000b0aea36'}

    path_data = './data'

    for c_family in c_families:

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
            tf.keras.backend.clear_session()

            print(f'loading model {model_name}: {models[model_name]}')
            m = load_model_helper(path_models, models[model_name])

            y = m.predict(x_test, batch_size=32, verbose=True)

            labels_pred = thres(y, thr=0.5)
            confusion_matr = confusion_matrix(y_test, labels_pred)
            confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

            print('Threshold: 0.5')
            print('Confusion matrix:')
            print(confusion_matr)

            print('Normalized confusion matrix:')
            print(confusion_matr_normalized)

            labels_pred = thres(y, thr=0.9)
            confusion_matr = confusion_matrix(y_test, labels_pred)
            confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

            print('Threshold: 0.9')
            print('Confusion matrix:')
            print(confusion_matr)

            print('Normalized confusion matrix:')
            print(confusion_matr_normalized)

            fpr, tpr, thresholds = roc_curve(y_test, y)
            roc_auc = auc(fpr, tpr)

            fig = plt.figure()
            lw = 1.6
            ax = fig.add_subplot(121)
            ax.plot(fpr, tpr, lw=lw, label=f'{model_name} curve (area = {roc_auc:.5f})')
            ax.plot([0, 1], [0, 1], color='#333333', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            ax.grid(True)

            plt.show()
