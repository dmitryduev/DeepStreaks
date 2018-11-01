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
    model_names = {'rb': 'ResNet50_rb_50e_20181031_150155.h5',
                   'sl': 'ResNet50_sl_20e_20181024_163759.h5'}

    models = dict()
    print('loading models...')
    models['rb'] = load_model(os.path.join(path_models, model_names['rb']))
    # models['sl'] = load_model(os.path.join(path_models, model_names['sl']))
    print('done')

    # print(models['rb'].summary())

    model_input_shape = models['rb'].input_shape[1:3]

    path_streaks_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/' + \
                        'reals_20180901_20181031'

    path_streak_stamps = glob.glob(os.path.join(path_streaks_base, '*.jpg'))

    scores = []

    for ip, path_streak_stamp in enumerate(path_streak_stamps):
        # print(path_streak_stamp)
        x = np.array(ImageOps.grayscale(Image.open(path_streak_stamp)).resize(model_input_shape,
                                                                              Image.BILINEAR)) / 255.
        x = np.expand_dims(x, 2)
        x = np.expand_dims(x, 0)

        # tic = time.time()
        rb = float(models['rb'].predict(x, batch_size=1)[0][0])

        if rb < 0.99:
            # print(f'____{path_streak_stamp}: {rb}')
            copyfile(path_streak_stamp, path_streak_stamp.replace('reals_20180901_20181031',
                                                                  'reals_20180901_20181031_rb_lt_0.99'))

        print(f'{ip+1}/{len(path_streak_stamps)} {path_streak_stamp}: {rb}')
        scores.append(rb)

    # plot hist
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.hist(scores, bins=100)
    plt.grid()

    plt.show()
