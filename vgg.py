import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import glob
import os
from PIL import Image

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def load_dataset():
    path_long_streaks = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/long-streaks'

    # df = pd.read_csv('/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/ztf-astreaks-classifications.csv')
    # data = [eval(s.replace('null', 'None')) for s in df.subject_data]
    # pattern = r'(strkid\d+._pid\d+._scimref.jpg)'
    # fss = [re.search(pattern, s.replace('null', 'None')).groups(0)[0] for s in df.subject_data]

    long_streaks = glob.glob(os.path.join(path_long_streaks, '*.jpg'))

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
        7: "Skip (Includes 'Not Sure' and seemingly 'Blank Images'"
    }

    for ls in long_streaks:
        # resize and normalize:
        image = np.expand_dims(np.array(Image.open(ls).resize((144, 144), Image.BILINEAR)) / 255., 2)
        x.append(image)
        image_class = np.zeros(8)
        image_class[1] = 1
        y.append(image_class)
        # raise Exception()

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, classes


def VGGModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def main():
    # load data. resize here
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    # image shape:
    image_shape = X_train.shape[1:]
    print(image_shape)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    # build model
    model = VGGModel(image_shape)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    model.fit(x=X_train, y=Y_train, epochs=1, batch_size=16)

    preds = model.evaluate(x=X_test, y=Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    print(model.summary())

    plot_model(model, to_file='VGG_model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


if __name__ == '__main__':

    main()
