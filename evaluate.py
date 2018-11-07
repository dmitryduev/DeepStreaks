import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import ImageOps, Image
from keras.models import load_model
import keras.backend as K
from shutil import copyfile
# K.set_image_data_format('channels_last')
K.set_learning_phase(0)

if __name__ == '__main__':

    path_models = '/Users/dmitryduev/_caltech/python/deep-asteroids/service/models'
    # model_names = {'rb': 'ResNet50_rb_50e_20181031_150155.h5',
    #                'sl': 'ResNet50_sl_20e_20181024_163759.h5'}

    # model_names = {'rb_resnet50': 'ResNet50_rb_50e_20181102_132634.h5',
    #                'sl_resnet50': 'ResNet50_sl_50e_20181103_012034.h5',
    #                'rb_vgg6': 'VGG6_rb_78e_20181103_001536.h5',
    #                'sl_vgg6': 'VGG6_sl_68e_20181102_234533.h5'}

    model_names = {'rb_vgg6': 'VGG6_rb_78e_20181103_001536.h5',
                   'sl_vgg6': 'VGG6_sl_68e_20181102_234533.h5',
                   'kd_vgg6': 'VGG6_kd_63e_20181106_152448.h5',
                   'kd_resnet50': 'ResNet50_kd_32e_20181106_171833.h5'}

    print('loading models...')
    models = {m: load_model(os.path.join(path_models, model_names[m])) for m in model_names.keys()}
    print(models['rb_vgg6'].summary())
    print('done')

    # print(models['rb_vgg6'].summary())

    model_input_shape = models['rb_vgg6'].input_shape[1:3]

    path_streaks_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/' + \
                        'reals_20181101_20181106'

    path_streak_stamps = glob.glob(os.path.join(path_streaks_base, '*.jpg'))

    scores = {m: [] for m in model_names.keys()}

    for ip, path_streak_stamp in enumerate(path_streak_stamps):
        # print(path_streak_stamp)
        x = np.array(ImageOps.grayscale(Image.open(path_streak_stamp)).resize(model_input_shape,
                                                                              Image.BILINEAR)) / 255.
        x = np.expand_dims(x, 2)
        x = np.expand_dims(x, 0)

        x_scores = dict()

        for m in model_names.keys():
            # tic = time.time()
            score = float(models[m].predict(x, batch_size=1)[0][0])

            # if score < 0.99:
            #     print(f'____{path_streak_stamp}: {score}')
            #     copyfile(path_streak_stamp, path_streak_stamp.replace('reals_20180901_20181031',
            #                                                           'reals_20180901_20181031_rb_lt_0.99'))

            scores[m].append(score)
            x_scores[m] = score
        print(f'{ip+1}/{len(path_streak_stamps)} {path_streak_stamp}:')
        print(f'{x_scores}\n')

    print(f'{scores}')

    # plot hist
    # fig = plt.figure()
    #
    # ax = fig.add_subplot(111)
    # ax.hist(scores, bins=100)
    # plt.grid()
    #
    # plt.show()
