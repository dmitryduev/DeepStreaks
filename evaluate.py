import os
import glob
import numpy as np
import time
from PIL import ImageOps, Image
from keras.models import load_model
import keras.backend as K
# K.set_image_data_format('channels_last')
K.set_learning_phase(0)

if __name__ == '__main__':

    path_models = '/Users/dmitryduev/_caltech/python/deep-asteroids/service/models'
    model_names = {'rb': 'ResNet50_rb_20e_20180914_122326.h5',
                   'sl': 'ResNet50_sl_20e_20180914_120303.h5'}

    models = dict()
    print('loading models...')
    models['rb'] = load_model(os.path.join(path_models, model_names['rb']))
    # models['sl'] = load_model(os.path.join(path_models, model_names['sl']))
    print('done')

    # print(models['rb'].summary())

    model_input_shape = models['rb'].input_shape[1:3]

    # path_streak_stamp = '/Users/dmitryduev/_caltech/python/deep-asteroids/_tmp/stamps/stamps_20180927/' + \
    #     'strkid6341069004150002_pid634106900415_scimref.jpg'

    path_streaks_base = '/Users/dmitryduev/_caltech/python/deep-asteroids/data/' + \
                        '5b96af9c0354c9000b0aea36/5b96ecf05ec848000c70a870.20180914_165152'

    path_streak_stamps = glob.glob(os.path.join(path_streaks_base, '*.jpg'))

    for path_streak_stamp in path_streak_stamps:
        print(path_streak_stamp)
        x = np.array(ImageOps.grayscale(Image.open(path_streak_stamp)).resize(model_input_shape,
                                                                              Image.BILINEAR)) / 255.
        x = np.expand_dims(x, 2)
        x = np.expand_dims(x, 0)

        # tic = time.time()
        rb = float(models['rb'].predict(x, batch_size=1)[0][0])

        print(rb)
