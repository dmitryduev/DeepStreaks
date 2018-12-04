from sklearn.metrics import confusion_matrix
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, concatenate, Dropout
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam, SGD
import datetime
import numpy as np

from utils import load_data
import keras.backend as K


def shallow_inception(input_shape=(144, 144, 1), n_classes: int=1):
    # Define the input as a tensor with shape input_shape
    x_input = Input(shape=input_shape)

    # Zero-Padding
    # x = ZeroPadding2D((3, 3))(x_input)

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x_input)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x_input)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x_input)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    incept = concatenate([tower_1, tower_2, tower_3], axis=1)

    x = Flatten()(incept)

    # hidden FC layer
    # x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    x = Dense(n_classes, activation=activation, name='fcOUT')(x)

    # Create model
    model = Model(inputs=x_input, outputs=x, name='ShallowInception')

    return model


def shallow_vgg(input_shape=(144, 144, 1), n_classes: int=1):

    # # batch norm momentum
    # batch_norm_momentum = 0.2

    model = Sequential(name='ShallowVGG')
    # input: 144x144 images with 1 channel -> (144, 144, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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


if __name__ == '__main__':
    K.clear_session()

    save_model = True

    ''' load data '''
    # streak / not streak? or with subclasses of bogus?
    binary_classification = True
    # binary_classification = False
    n_classes = 1 if binary_classification else 2
    loss = 'binary_crossentropy' if binary_classification else 'categorical_crossentropy'

    # load data
    # project_id = '5b96af9c0354c9000b0aea36'  # real vs bogus
    project_id = '5b99b2c6aec3c500103a14de'  # short vs long
    # project_id = '5be0ae7958830a0018821794'  # keep vs ditch

    X_train, Y_train, X_test, Y_test, classes = load_data(path='./data',
                                                          project_id=project_id,
                                                          binary=binary_classification,
                                                          resize=(144, 144),
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
    # model = shallow_inception(input_shape=image_shape, n_classes=n_classes)
    # model = shallow_vgg(input_shape=image_shape, n_classes=n_classes)
    model = vgg6(input_shape=image_shape, n_classes=n_classes)
    # model = vgg4(input_shape=image_shape, n_classes=n_classes)

    # set up optimizer:
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0)
    # sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

    print(model.summary())

    tensorboard = TensorBoard(log_dir=f'./logs/{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}')

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)

    batch_size = 32

    # model.fit(X_train, Y_train, epochs=20, batch_size=batch_size, verbose=1, callbacks=[tensorboard])
    model.fit(X_train, Y_train, epochs=200, batch_size=batch_size, shuffle=True,
              class_weight={0: 1, 1: 1},
              validation_split=0.05,
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

    model_save_name = f'./{datetime.datetime.now().strftime(model.name + "_%Y%m%d_%H%M%S")}'
    if True:
        model_save_name_h5 = f'{model_save_name}.h5'
        model.save(model_save_name_h5)

    print(f'Batch size: {batch_size}')
    preds = model.predict(x=X_test, batch_size=batch_size)
    # print(preds)

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
        f.write(f'Loss = {test_loss:.4f}\n')
        f.write(f'Test Accuracy = {test_accuracy:.4f}\n')
        f.write('\nConfusion matrix:\n')
        f.write(str(confusion_matr))
        f.write('\nNormalized confusion matrix:\n')
        f.write(str(confusion_matr_normalized))
