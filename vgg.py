import numpy as np
from time import time
import datetime
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import glob
import os
from PIL import Image
import json

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


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


def VGGModel(input_shape, nf: tuple=(16, 32), f: int=3, s: int=1, nfc: int=128, n_classes: int=8):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset
    f -- filter size
    s -- stride

    padding is always 'same'

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    nf1, nf2 = nf

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(nf1, (f, f), strides=(s, s), padding='same', name='conv0')(X_input)
    # X = Conv2D(nf1, (f, f), strides=(s, s), padding='same', name='conv0', data_format=K.image_data_format())(X_input)
    X = BatchNormalization(axis=-1, name='bn0')(X, training=1)
    # X = BatchNormalization(axis=-1, name='bn0')(X)
    X = Activation('relu')(X)
    # X = Activation('sigmoid')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool0')(X)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(nf2, (f, f), strides=(s, s), padding='same', name='conv1')(X)
    # X = Conv2D(nf2, (f, f), strides=(s, s), padding='same', name='conv1', data_format=K.image_data_format())(X)
    X = BatchNormalization(axis=-1, name='bn1')(X, training=1)
    # X = BatchNormalization(axis=-1, name='bn1')(X)
    X = Activation('relu')(X)
    # X = Activation('sigmoid')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1')(X)

    # FLATTEN X (means convert it to a vector)
    X = Flatten()(X)

    # FULLYCONNECTED
    X = Dense(nfc, activation='sigmoid', name='fc2')(X)

    # FULLYCONNECTED
    # X = Dense(nfc, activation='sigmoid', name='fc3')(X)

    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    X = Dense(n_classes, activation=activation, name='fcOUT', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='vgg_model_v1')

    return model


def VGGModel_v2(input_shape, nf: tuple=(16, 32, 64), f: int=3, s: int=1, nfc: tuple=(128,), n_classes: int=8):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset
    nf -- number of filters in conv blocks
    f -- filter size
    s -- stride
    nf -- number of neurons in FC layers

    padding is always 'same'

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    ''' first convolutional block: [CONV] -> [BATCH_NORM] -> [RELU] -> [MAXPOOL] '''

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(nf[0], (f, f), strides=(s, s), padding='same', name='conv0')(X_input)
    X = BatchNormalization(axis=-1, name='bn0')(X, training=1)
    X = Activation('relu')(X)
    # X = Activation('sigmoid')(X)
    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool0')(X)

    ''' convolutional blocks: [CONV] -> [BATCH_NORM] -> [RELU] -> [MAXPOOL] '''
    for i in range(1, len(nf)):
        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(nf[i], (f, f), strides=(s, s), padding='same', name=f'conv{i}')(X)
        X = BatchNormalization(axis=-1, name=f'bn{i}')(X, training=1)
        X = Activation('relu')(X)
        # X = Activation('sigmoid')(X)
        # MAXPOOL
        X = MaxPooling2D((2, 2), strides=(2, 2), name=f'max_pool{i}')(X)

    ''' FLATTEN X (means convert it to a vector) '''
    X = Flatten()(X)

    ''' FULLYCONNECTED layers '''
    for i, nfc_i in enumerate(nfc):
        X = Dense(nfc_i, activation='sigmoid', name=f'fc{i+len(nf)}')(X)

    ''' FULLYCONNECTED output layer '''
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    X = Dense(n_classes, activation=activation, name='fcOUT', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='vgg_model_v2')

    return model


def main():
    K.clear_session()

    # streak / not streak? or with subclasses of bogus?
    # binary_classification = True
    binary_classification = False
    n_classes = 1 if binary_classification else 8
    n_fc = 32 if binary_classification else 128
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

    # build model
    # model = VGGModel(image_shape, n_classes=n_classes)
    # model = VGGModel(image_shape, nf=(16, 32), f=3, s=1, nfc=128, n_classes=n_classes)
    # model = VGGModel(image_shape, nf=(16, 32), f=3, s=1, nfc=32, n_classes=n_classes)
    # model = VGGModel(image_shape, nf=(16, 32), f=3, s=1, nfc=n_fc, n_classes=n_classes)

    model = VGGModel_v2(image_shape, nf=(16, 32, 64), f=3, s=1, nfc=(256,), n_classes=n_classes)

    # set up optimizer:
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f'./logs/{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}')

    batch_size = 32

    model.fit(x=X_train, y=Y_train, epochs=10, batch_size=batch_size, verbose=1, callbacks=[tensorboard])

    # preds = model.evaluate(x=X_train, y=Y_train)
    preds = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
    # preds = model.evaluate(x=X_test, y=Y_test, batch_size=X_test.shape[0])
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    # preds = model.evaluate(x=X_test, y=Y_test)
    # print("Loss = " + str(preds[0]))
    # print("Test Accuracy = " + str(preds[1]))

    # print(model.summary())

    model.save(f'./{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}.h5')

    # plot_model(model, to_file=f'{model.name}.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))


if __name__ == '__main__':

    main()
