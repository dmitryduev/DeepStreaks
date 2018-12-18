import os
import glob
import time
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
# import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def find_files(root_dir, extension: str='jpg'):
    for dir_name, subdir_list, file_list in os.walk(root_dir, followlinks=True):
        for f_name in file_list:
            if f_name.endswith(f'.{extension}'):
                yield os.path.join(dir_name, f_name)


def load_data_predict(path_images=(), grayscale: bool=True,
                      resize: tuple=(144, 144), verbose: bool=True):

    num_images = len(path_images)
    num_channels = 1 if grayscale else 3

    # allocate:
    data = np.zeros((num_images, *resize, num_channels))
    img_ids = np.zeros(num_images, dtype=object)

    for ii, path_image in enumerate(path_images):
        image_basename = os.path.basename(path_image)
        img_id = image_basename.split('_scimref.jpg')[0]
        img_ids[ii] = img_id

        if grayscale:
            img = np.array(ImageOps.grayscale(Image.open(path_image)).resize(resize, Image.BILINEAR)) / 255.
            img = np.expand_dims(img, 2)
        else:
            img = ImageOps.grayscale(Image.open(path_image)).resize(resize, Image.BILINEAR)
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = np.array(rgbimg) / 255.

        data[ii, :] = img

    return data, img_ids


if __name__ == '__main__':

    # path_data = '/Users/dmitryduev/_caltech/python/deep-asteroids/data/' + \
    #             '5b96af9c0354c9000b0aea36/5b9b5de7dc6dc50010f3a0f2.20180914_165216'

    path_data = '/data/streaks/stamps/stamps_20181215'

    path_streaks = glob.glob(os.path.join(path_data, '*.jpg'))

    # files_streaks = (os.path.basename(p) for p in glob.glob(os.path.join(path_data, '*.jpg')))

    print('loading image data')
    tic = time.time()
    images, image_ids = load_data_predict(path_images=tuple(find_files(path_data)))
    toc = time.time()
    print(images.shape)
    print(f'done. loaded {len(image_ids)} images, which took {toc-tic} seconds.')

    # path_model = '/Users/dmitryduev/_caltech/python/deep-asteroids/service/models/VGG6_rb_78e_20181103_001536.h5'
    path_model = '/app/models/5b96af9c0354c9000b0aea36_VGG6_20181207_151757.h5'

    print('loading model')
    # model = tf.keras.models.load_model(path_model)
    tic = time.time()
    model = load_model(path_model)
    toc = time.time()
    print(f'done. took {toc-tic} seconds')

    model_input_shape = model.input_shape[1:3]

    # Compute ML scores one by one:
    tic = time.time()
    for path_streak in path_streaks:
        x = np.array(ImageOps.grayscale(Image.open(path_streaks[0])).resize(model_input_shape, Image.BILINEAR)) / 255.
        x = np.expand_dims(x, 2)
        x = np.expand_dims(x, 0)

        score = model.predict(x)
        # print(os.path.basename(path_streaks[0]), score)

    toc = time.time()
    print(f'running prediction one by one took {toc-tic} seconds.')

    batch_size = 32
    tic = time.time()
    scores = model.predict(images, batch_size=batch_size)
    toc = time.time()
    print(f'running prediction with batch_size={batch_size} took {toc-tic} seconds.')
