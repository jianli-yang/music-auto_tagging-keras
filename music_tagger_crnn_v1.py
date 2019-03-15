# -*- coding: utf-8 -*-
'''MusicTaggerCRNN model for Keras.

Code by github.com/keunwoochoi.

# Reference:

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

'''
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from common import load_track, GENRES
import sys

BATCH_SIZE = 32
EPOCH_COUNT = 100


def data_generator(data, targets, batch_size):
    while True:
      cnt = 0
      xx = []
      yy = []
      for i in range(0, len(data)):

          tmpx, _, code = load_track(data[i], (1536, 128), True)
          if code != 0:
              continue

          tmpx = tmpx.T ## trans
          # tmpx = tmpx * 1.0 / tmpx.max()  ## 归一化
          tmpy = targets[i]

          xx.append(tmpx)
          yy.append(tmpy)
          cnt += 1
          if cnt >= batch_size:

            ret = (np.array(xx).reshape([-1, 128, 1536, 1]), np.array(yy).reshape(-1, len(GENRES)))
            cnt = 0
            xx = []
            yy = []
            yield ret


def test_split(track_paths, y):
    # track_paths = data['track_paths']
    # y = data["y"]

    indexs = [i for i in range(0, len(y))]
    np.random.shuffle(indexs)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    llen = int(len(y) * 0.8)
    for i in range(0, len(indexs)):
        if i < llen:
            x_train.append(track_paths[indexs[i]])
            y_train.append(y[indexs[i]])
        else:
            x_val.append(track_paths[indexs[i]])
            y_val.append(y[indexs[i]])

    return x_train, y_train, x_val, y_val


def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.


    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 128, 1536)
    else:
        input_shape = (128, 1536, 1) #tensorflow

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)
    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((12, 64))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    if include_top:
        x = Dense(8, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    return model


if __name__ == '__main__':
    model_path = sys.argv[1]
    data = np.load("/data/jianli.yang/music-raw-data-8000/mels/data.pkl.npy")
    data = data.tolist()
    x_train, y_train, x_val, y_val = test_split(data["track_paths"], data["y"])
    model = MusicTaggerCRNN(weights='msd')
    opt = Adam(lr=0.01)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
    model.summary()
    print('Training...')

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

    model.fit_generator(
       generator=data_generator(x_train, y_train, BATCH_SIZE), epochs=EPOCH_COUNT, steps_per_epoch=200, validation_steps=50,
        validation_data=data_generator(x_val, y_val, BATCH_SIZE), verbose=1, callbacks=[
            ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_acc', verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001,
                verbose=1
            )
        ]
    )
