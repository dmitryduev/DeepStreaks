import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse

from sklearn.metrics import confusion_matrix
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
                         AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, \
                         concatenate, Dropout
from keras.models import Model, Sequential, load_model
# from keras.applications.densenet import DenseNet121
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_uniform
import keras.utils as keras_utils
from keras.preprocessing.image import ImageDataGenerator
import datetime
import numpy as np

from utils import load_data
import keras.backend as K


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
    # batch norm momentum
    batch_norm_momentum = 0.2

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
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
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
    # batch norm momentum
    batch_norm_momentum = 0.2

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
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut, training=1)
    X_shortcut = BatchNormalization(axis=-1, momentum=0.1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(144, 144, 1), n_classes: int=1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    n_classes -- integer, number of classes. if = 1, sigmoid is used in the output layer; softmax otherwise

    Returns:
    model -- a Model() instance in Keras
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name='bn_conv1')(X, training=1)
    X = BatchNormalization(axis=-1, momentum=batch_norm_momentum, name='bn_conv1')(X)
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


def vgg6(input_shape=(144, 144, 1), n_classes: int=1):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """

    # # batch norm momentum
    # batch_norm_momentum = 0.2

    model = Sequential(name='VGG6')
    # input: 144x144 images with 1 channel -> (144, 144, 1) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    # model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation, name='fc_out'))

    return model


def vgg4(input_shape=(144, 144, 1), n_classes: int=1):

    # # batch norm momentum
    # batch_norm_momentum = 0.2

    model = Sequential(name='VGG4')
    # input: 144x144 images with 1 channel -> (144, 144, 1) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum))
    # model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation))

    return model


def fc4(input_shape=(144, 144, 1), n_classes: int=1):

    # # batch norm momentum
    # batch_norm_momentum = 0.2

    model = Sequential(name='FC4')
    # input: 144x144 images with 1 channel -> (144, 144, 1) tensors.
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation))

    return model


def dense_block(x, blocks, name):
    """A dense block for DenseNet.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block for DenseNet.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis,
                                  epsilon=1.001e-5,
                                  momentum=batch_norm_momentum,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    # batch norm momentum
    batch_norm_momentum = 0.2

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   momentum=batch_norm_momentum,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   momentum=batch_norm_momentum,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(input_shape=(144, 144, 1), n_classes: int=1, include_top: bool=True):

    # batch norm momentum
    batch_norm_momentum = 0.2

    # densenet121
    blocks = [6, 12, 24, 16]
    # densenet169
    # blocks = [6, 12, 32, 32]
    # densenet201
    # blocks = [6, 12, 48, 32]

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(X_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, momentum=batch_norm_momentum, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, momentum=batch_norm_momentum, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        # output layer
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        activation = 'sigmoid' if n_classes == 1 else 'softmax'
        x = layers.Dense(n_classes, activation=activation, name='fc_out')(x)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(X_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(X_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(X_input, x, name='densenet201')
    else:
        model = Model(X_input, x, name='densenet')

    return model


def DenseNet_imagenet(input_shape=(144, 144, 1), n_classes: int=1, freeze_weights: bool=False):

    BASE_WEIGTHS_PATH = (
        'https://github.com/keras-team/keras-applications/'
        'releases/download/densenet/')
    DENSENET121_WEIGHT_PATH = (
            BASE_WEIGTHS_PATH +
            'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
    DENSENET121_WEIGHT_PATH_NO_TOP = (
            BASE_WEIGTHS_PATH +
            'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    DENSENET169_WEIGHT_PATH = (
            BASE_WEIGTHS_PATH +
            'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
    DENSENET169_WEIGHT_PATH_NO_TOP = (
            BASE_WEIGTHS_PATH +
            'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
    DENSENET201_WEIGHT_PATH = (
            BASE_WEIGTHS_PATH +
            'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
    DENSENET201_WEIGHT_PATH_NO_TOP = (
            BASE_WEIGTHS_PATH +
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # densenet121
    blocks = [6, 12, 24, 16]
    # densenet169
    # blocks = [6, 12, 32, 32]
    # densenet201
    # blocks = [6, 12, 48, 32]

    # base densenet to freeze
    base_model = DenseNet(input_shape=input_shape, n_classes=n_classes, include_top=False)

    # load imagenet weights:
    if blocks == [6, 12, 24, 16]:
        weights_path = keras_utils.get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET121_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='30ee3e1110167f948a6b9946edeeb738')
    elif blocks == [6, 12, 32, 32]:
        weights_path = keras_utils.get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET169_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
    elif blocks == [6, 12, 48, 32]:
        weights_path = keras_utils.get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET201_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
    else:
        raise Exception('Do not have such weights :(')

    base_model.load_weights(weights_path, skip_mismatch=True)

    x = base_model.output
    # add a global spatial average pooling layer
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # output FC layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    predictions = layers.Dense(n_classes, activation=activation, name='fc_out')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name='densenet121_imagenet')

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers
    if freeze_weights:
        for layer in base_model.layers:
            layer.trainable = False

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepStreaks')
    parser.add_argument('--project_id', type=str,
                        help='Zwickyverse project id. As of 20181204:\n' +
                             "\t'5b96af9c0354c9000b0aea36'  : real vs bogus" +
                             "\t'5b99b2c6aec3c500103a14de'  : short vs long" +
                             "\t'5be0ae7958830a0018821794'  : keep vs ditch" +
                             "\t'5c05bbdc826480000a95c0bf'  : one shot",
                        default='5c05bbdc826480000a95c0bf')
    parser.add_argument('--path_data', type=str,
                        help='Local path to data',
                        default='./data')
    parser.add_argument('--model', type=str,
                        help='Choose model to train: FC4, VGG4, VGG6, ResNet50, DenseNet121, DenseNet121_imagenet',
                        default='VGG6')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size',
                        default=32)
    parser.add_argument('--loss', type=str,
                        help='Loss function: binary_crossentropy or categorical_crossentropy',
                        default='binary_crossentropy')
    parser.add_argument('--optimizer', type=str,
                        help='Optimized to use: adam or sgd',
                        default='adam')
    parser.add_argument('--epochs', type=int,
                        help='Number of train epochs',
                        default=200)
    parser.add_argument('--patience', type=int,
                        help='Early stop training if no val_acc improvement after this many epochs',
                        default=150)
    parser.add_argument('--class_weight', action='store_true',
                        help='Weight training data by class depending on number of examples')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose')

    args = parser.parse_args()
    project_id = args.project_id
    path_data = args.path_data

    # known models:
    models = {'FC4': {'model': fc4, 'grayscale': True},
              'VGG4': {'model': vgg4, 'grayscale': True},
              'VGG6': {'model': vgg6, 'grayscale': True},
              'ResNet50': {'model': ResNet50, 'grayscale': True},
              'DenseNet121': {'model': DenseNet, 'grayscale': True},
              'DenseNet121_imagenet': {'model': DenseNet_imagenet, 'grayscale': False}
              }
    assert args.model in models, f'Unknown model: {args.model}'
    grayscale = models[args.model]['grayscale']

    K.clear_session()

    save_model = True

    ''' load data '''
    loss = args.loss
    binary_classification = True if loss == 'binary_crossentropy' else False
    n_classes = 1 if binary_classification else 2

    # load data
    X_train, Y_train, X_test, Y_test, classes = load_data(path=path_data,
                                                          project_id=project_id,
                                                          binary=binary_classification,
                                                          grayscale=grayscale,
                                                          resize=(144, 144),
                                                          test_size=0.1,
                                                          verbose=args.verbose,
                                                          random_state=42)

    # training data weights
    if args.class_weight:
        # weight data class depending on number of examples?
        if not binary_classification:
            num_training_examples_per_class = np.sum(Y_train, axis=0)
        else:
            num_training_examples_per_class = np.array([len(Y_train) - np.sum(Y_train), np.sum(Y_train)])

        assert 0 not in num_training_examples_per_class, 'found class without any examples!'

        # fewer examples -- larger weight
        weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
        normalized_weight = weights / np.max(weights)

        class_weight = {i: w for i, w in enumerate(normalized_weight)}

    else:
        class_weight = {i: 1 for i, _ in enumerate(classes.keys())}

    print(f'Class weights: {class_weight}\n')

    # image shape:
    image_shape = X_train.shape[1:]
    print('Input image shape:', image_shape)

    print("Number of training examples = " + str(X_train.shape[0]))
    print("Number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    ''' build model '''
    model = models[args.model]['model'](input_shape=image_shape, n_classes=n_classes)
    # model = vgg4(input_shape=image_shape, n_classes=n_classes)

    # set up optimizer:
    if args.optimizer == 'adam':
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif args.optimizer == 'sgd':
        optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        # optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0)
    else:
        print('Could not recognize optimizer, using Adam')
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

    # print(model.summary())

    model_name = f'{project_id}_{model.name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    tensorboard = TensorBoard(log_dir=f'./logs/{model_name}')

    patience = args.patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    batch_size = args.batch_size

    epochs = args.epochs

    # training without data augmentation
    # model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
    #           class_weight=class_weight,
    #           validation_split=0.05,
    #           verbose=1, callbacks=[tensorboard, early_stopping])

    # training with data augmentation:
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, validation_split=0.05)

    training_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='training')
    validation_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='validation')

    model.fit_generator(training_generator,
                        steps_per_epoch=len(X_train) // batch_size,
                        validation_data=validation_generator,
                        validation_steps=(len(X_train)*0.05) // batch_size,
                        class_weight=class_weight,
                        epochs=epochs,
                        verbose=1, callbacks=[tensorboard, early_stopping])

    # print('Evaluating on training set to check for BatchNorm behavior:')
    # preds = model.evaluate(X_train, Y_train, batch_size=batch_size)
    # print("Loss in prediction mode = " + str(preds[0]))
    # print("Training Accuracy in prediction mode = " + str(preds[1]))

    print('Evaluating on test set')
    preds = model.evaluate(X_test, Y_test, batch_size=batch_size)
    test_loss = preds[0]
    test_accuracy = preds[1]
    print("Loss = " + str(test_loss))
    print("Test Accuracy = " + str(test_accuracy))

    # save the full model [h5] and also separately weights [h5] and architecture [json]:
    model_save_name = f'./{model_name}'
    if True:
        model_save_name_h5 = f'{model_save_name}.h5'
        model.save(model_save_name_h5)

        model.save_weights(f'{model_save_name}.weights.h5')
        model_json = model.to_json()
        with open(f'{model_save_name}.architecture.json', 'w') as json_file:
            json_file.write(model_json)

    print(f'Batch size: {batch_size}')
    preds = model.predict(x=X_test, batch_size=batch_size)

    # round probs to nearest int (0 or 1)
    labels_pred = np.rint(preds)
    confusion_matr = confusion_matrix(Y_test, labels_pred)
    confusion_matr_normalized = confusion_matr.astype('float') / confusion_matr.sum(axis=1)[:, np.newaxis]

    print('Confusion matrix:')
    print(confusion_matr)

    print('Normalized confusion matrix:')
    print(confusion_matr_normalized)

    # save test loss/accuracy and confusion matrices in a text file:
    with open(f'./{model_save_name}.txt', 'w') as f:
        f.write(f'Input image shape: {str(image_shape)}\n')

        f.write(f'Number of training examples = {str(X_train.shape[0])}\n')
        f.write(f'Number of test examples = {str(X_test.shape[0])}\n')
        f.write(f'X_train shape: {str(X_train.shape)}\n')
        f.write(f'Y_train shape: {str(Y_train.shape)}\n')
        f.write(f'X_test shape: {str(X_test.shape)}\n')
        f.write(f'Y_test shape: {str(Y_test.shape)}\n')

        # f.write('\nModel summary:\n')
        # f.write(str(model.summary()))

        f.write(f'\nLoss = {test_loss:.4f}\n')
        f.write(f'Test Accuracy = {test_accuracy:.4f}\n')
        f.write('\nConfusion matrix:\n')
        f.write(str(confusion_matr))
        f.write('\nNormalized confusion matrix:\n')
        f.write(str(confusion_matr_normalized))
