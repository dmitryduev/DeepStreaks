import numpy as np
import glob
import os
import time
import datetime
from PIL import Image, ImageOps
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


def load_data(path: str='./data', project_id: str=None, binary: bool=True, resize: tuple=(144, 144), test_size=0.1):

    # data:
    x = []
    # classifications:
    y = []

    # get json file with project metadata
    project_meta_json = os.path.join(path, f'{project_id}.json')
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

    path_project = os.path.join(path, project_id)

    for dataset_id in project_meta['datasets']:
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

                if os.path.exists(image_path):
                    # the assumption is that images are grayscale
                    img = np.array(ImageOps.grayscale(Image.open(image_path)).resize(resize, Image.BILINEAR)) / 255.
                    img = np.expand_dims(img, 2)
                    x.append(img)

                    image_class = classes[v[0]]
                    y.append(image_class)

    # numpy-fy and split to test/train

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    # check statistics on different classes
    if not binary:
        print('\n')
        for i in classes.keys():
            print(f'{i}:', np.sum(y[:, i]))
        print('\n')
    else:
        print('\n')
        cs = list(classes.keys())
        print(f'{cs[0]}:', len(y) - np.sum(y))
        print(f'{cs[1]}:', np.sum(y))
        print('\n')

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return x_train, y_train, x_test, y_test, classes


def load_data_custom(path: str = './data', project_id: str = None, binary: bool = True, resize: tuple = (144, 144),
                     test_size=0.1):
    # data:
    x = []
    # classifications:
    y = []

    # hand-picked test data:
    x_test = []
    y_test = []

    # get json file with project metadata
    project_meta_json = os.path.join(path, f'{project_id}.json')
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

    path_project = os.path.join(path, project_id)

    # FIXME:
    for dataset_id in project_meta['datasets'][:3]:
        print(f'Loading dataset {dataset_id}')

        dataset_json = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*.json')))[-1]
        with open(dataset_json) as f:
            classifications = json.load(f)
        # print(classifications)

        path_dataset = [p for p in sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*'))) if
                        os.path.isdir(p)][-1]
        # print(path_dataset)

        nnn = 1
        for k, v in classifications.items():
            image_class = classes[v[0]]
            if dataset_id == '5b96ecf05ec848000c70a870' and image_class == 1 and nnn <= 50:
                # FIXME: use streak examples from Zooniverse as test cases
                y_test.append(image_class)

                # resize and normalize:
                image_path = os.path.join(path_dataset, k)
                # the assumption is that images are grayscale
                image = np.expand_dims(np.array(ImageOps.grayscale(Image.open(image_path)).resize(resize,
                                                                                                  Image.BILINEAR)) / 255.,
                                       2)
                x_test.append(image)

                nnn += 1

            else:
                y.append(image_class)

                # resize and normalize:
                image_path = os.path.join(path_dataset, k)
                # the assumption is that images are grayscale
                image = np.expand_dims(np.array(ImageOps.grayscale(Image.open(image_path)).resize(resize,
                                                                                                  Image.BILINEAR)) / 255.,
                                       2)
                x.append(image)

    # numpy-fy and split to test/train

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    # check statistics on different classes
    if not binary:
        print('\n')
        for i in classes.keys():
            print(f'{i}:', np.sum(y[:, i]))
        print('\n')
    else:
        print('\n')
        cs = list(classes.keys())
        print(f'{cs[0]}:', len(y) - np.sum(y))
        print(f'{cs[1]}:', np.sum(y))
        print('\n')

    # # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # FIXME:
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train, x_test, y_train, y_test = x, x_test, y, y_test

    # add
    x_train, x_test_custom, y_train, y_test_custom = train_test_split(x_train, y_train, test_size=test_size)
    # print(x_test.shape, x_test_custom.shape)
    x_test = np.concatenate((x_test_custom, x_test))
    y_test = np.concatenate((y_test_custom, y_test))

    return x_train, y_train, x_test, y_test, classes


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
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)

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
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut, training=1)
    # X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(144, 144, 1), n_classes: int=8):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    n_classes -- integer, number of classes

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
    # X = BatchNormalization(axis=-1, name='bn_conv1')(X)
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
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    X = Dense(n_classes, activation=activation, name='fcOUT', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


if __name__ == '__main__':
    K.clear_session()

    ''' load data '''
    # streak / not streak? or with subclasses of bogus?
    binary_classification = True
    # binary_classification = False
    n_classes = 1 if binary_classification else 8
    # n_fc = 32 if binary_classification else 128
    loss = 'binary_crossentropy' if binary_classification else 'categorical_crossentropy'

    # load data
    project_id = '5b96af9c0354c9000b0aea36'  # real vs bogus
    # project_id = '5b99b2c6aec3c500103a14de'  # short vs long

    # X_train, Y_train, X_test, Y_test, classes = load_data_custom(path='./data',
    #                                                              project_id='5b96af9c0354c9000b0aea36',
    #                                                              binary=binary_classification,
    #                                                              test_size=0.05)
    X_train, Y_train, X_test, Y_test, classes = load_data(path='./data',
                                                          project_id=project_id,
                                                          binary=binary_classification,
                                                          test_size=0.1)

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
    model = ResNet50(input_shape=image_shape, n_classes=n_classes)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    # model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f'./logs/{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}')

    batch_size = 32

    model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, verbose=1, callbacks=[tensorboard])

    # turn off learning phase (beware BatchNormalization!)
    # K.set_learning_phase(0)

    # preds = model.evaluate(X_test, Y_test, batch_size=batch_size)
    preds = model.evaluate(X_test, Y_test, batch_size=1)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    if False:
        model.save(f'./{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}.h5')

    # preds = model.predict(x=X_test, batch_size=batch_size)
    preds = model.predict(x=X_test, batch_size=1)
    # print(preds)

    # round probs to nearest int (0 or 1)
    labels_pred = np.rint(preds)
    confusion_matrix = confusion_matrix(Y_test, labels_pred)
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    print('Confusion matrix:')
    print(confusion_matrix)

    print('Normalized confusion matrix:')
    print(confusion_matrix_normalized)

    # for ip, pred in enumerate(preds):
    #     print(pred[0])
    #     im = Image.fromarray((X_test[ip, :, :, 0] * 255).astype('uint8'))
    #     im.show()
    #     input()

    ''' test predictions '''
    path_streak_stamps = glob.glob(os.path.join('./data', project_id, '5b96ecf05ec848000c70a870.20180914_165152',
                                                '*.jpg'))
    model_input_shape = model.input_shape[1:3]

    for path_streak_stamp in path_streak_stamps[:5]:
        print(path_streak_stamp)
        x = np.array(ImageOps.grayscale(Image.open(path_streak_stamp)).resize(model_input_shape,
                                                                              Image.BILINEAR)) / 255.
        x = np.expand_dims(x, 2)
        x = np.expand_dims(x, 0)

        tic = time.time()
        rb = float(model.predict(x, batch_size=1)[0][0])
        toc = time.time()

        print(rb, f'took {toc-tic} seconds')
