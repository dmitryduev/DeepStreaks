import numpy as np
import glob
import os
import datetime
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
# %matplotlib inline

import keras.backend as K
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)


def load_dataset(binary: bool=False, test_size=0.1):
    """

    :return:
    """
    # path = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/'
    path = './data-raw/'

    # cut-outs:
    x = []
    # classifications:
    y = []

    # 08/21/2018: 8 possible classes:
    classes = {
        0: "Plausible Asteroid (short streak)",
        1: "Satellite (long streak - could be partially masked)",
        2: "Masked bright star",
        3: "Dementors and ghosts",
        4: "Cosmic rays",
        5: "Yin-Yang (multiple badly subtracted stars)",
        6: "Satellite flashes",
        7: "Skip (Includes 'Not Sure' and seemingly 'Blank Images')"
    }

    ''' Long streaks from Quanzhi '''
    path_long_streaks = os.path.join(path, 'long-streaks')

    long_streaks = glob.glob(os.path.join(path_long_streaks, '*.jpg'))

    for ls in long_streaks:
        # resize and normalize:
        image = np.expand_dims(np.array(Image.open(ls).resize((144, 144), Image.BILINEAR)) / 255., 2)
        x.append(image)

        if binary:
            # image_class = np.zeros(2)
            image_class = 1
        else:
            image_class = np.zeros(8)
            image_class[1] = 1

        y.append(image_class)
        # raise Exception()

    ''' Stuff from Zooniverse '''
    # TODO
    # get json file with classifications
    zoo_json = os.path.join(path, 'zooniverse.20180824_010749.json')
    with open(zoo_json) as f:
        zoo_classifications = json.load(f)

    path_zoo = os.path.join(path, 'zooniverse')

    zoos = glob.glob(os.path.join(path_zoo, '*.jpg'))

    for z in zoos:
        z_fname = os.path.split(z)[1]
        if z_fname in zoo_classifications:
            # resize and normalize:
            image = np.expand_dims(np.array(Image.open(z).resize((144, 144), Image.BILINEAR)) / 255., 2)
            x.append(image)

            if binary:
                # image_class = np.zeros(2)
                index = list(classes.values()).index(zoo_classifications[z_fname][0])
                if index in (0, 1):
                    image_class = 1
                else:
                    image_class = 0
            else:
                image_class = np.zeros(8)
                image_class[list(classes.values()).index(zoo_classifications[z_fname][0])] = 1

            y.append(image_class)

    # numpy-fy and split to test/train

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    # check statistics on different classes
    print('\n')
    for i in classes.keys():
        print(f'{classes[i]}:', np.sum(y[:, i]))
    print('\n')

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, y_train, X_test, y_test, classes


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut, training=1)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name='bn_conv1')(X, training=1)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 512], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 1024], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 1024], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 1024], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


if __name__ == '__main__':
    K.clear_session()

    ''' load data '''

    # streak / not streak? or with subclasses of bogus?
    # binary_classification = True
    binary_classification = False
    n_classes = 1 if binary_classification else 8
    # n_fc = 32 if binary_classification else 128
    loss = 'binary_crossentropy' if binary_classification else 'categorical_crossentropy'

    # load data. resize here
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(binary=binary_classification,
                                                                                 test_size=0.1)

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    #
    Y_train = Y_train_orig
    Y_test = Y_test_orig

    # image shape:
    image_shape = X_train.shape[1:]
    print('image shape:', image_shape)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    ''' build model '''
    model = ResNet50(input_shape=(144, 144, 1), classes=n_classes)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f'./logs/{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}')

    batch_size = 32

    model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, verbose=1, callbacks=[tensorboard])

    preds = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

