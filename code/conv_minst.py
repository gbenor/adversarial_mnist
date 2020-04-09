import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model

from abc_model import AbcModel
from consts import num_classes
from utils import categorical_crossentropy_with_temp


class ConvMinst(AbcModel):
    ''' Build a simple MNIST classification CNN
        The network takes ~3 minutes to train on a normal laptop and reaches roughly 97% of accuracy
        Model structure: Conv, Conv, Max pooling, Dropout, Dense, Dense
    '''
    def build(self):
        activation = 'relu'
        # input image dimensions
        img_rows, img_cols, img_colors = 28, 28, 1

        model = keras.Sequential()
        model.add(layers.Conv2D(8, kernel_size=(3, 3), input_shape=(img_rows, img_cols, img_colors), activation=activation))
        model.add(layers.Conv2D(8, (3, 3), activation=activation))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation=activation))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes))
        # model.add(layers.Activation('softmax', name='y_pred'))
        self._model = model
        self.compile()


        print("finish build and compile conv_minst")

    def compile(self):
        def fn(correct, predicted):
            return categorical_crossentropy_with_temp(correct, predicted, self._temp)

        self._model.compile(loss=fn,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=[keras.metrics.CategoricalAccuracy()])

