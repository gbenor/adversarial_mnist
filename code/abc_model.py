from abc import ABC, abstractmethod
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import model_from_json

from adv_logger import logger


class AbcModel(ABC):
    def __init__(self, fname, temp=1):
        self._temp = temp
        self._fname = f"{fname}_temp_{temp}"
        self._fname_json = Path(f"{self._fname}.json")
        self._fname_weights = Path(f"{self._fname}.h5")

        if self.model_exists():
            print(f"loading model: {self._fname}")
            self.load()
        else:
            self.build()

    def name(self):
        return self._fname

    def model_exists(self):
        return (self._fname_json.exists() and self._fname_weights.exists())

    def model(self):
        return self._model

    def temp(self):
        return self._temp

    def add_softmax_layer(self):
        self._model.add(layers.Activation('softmax', name='y_pred'))

    def save(self):
        logger.info(f"save: {self._fname}")
        # Save JSON config to disk
        with self._fname_json.open("w") as json_file:
            json_file.write(self._model.to_json())
        # Save weights to disk
        self._model.save_weights(str(self._fname_weights))

    def load(self):
        logger.info(f"load: {self._fname}")
        # Reload the model from the 2 files we saved
        with self._fname_json.open("r") as json_file:
            json_config = json_file.read()
        self._model = model_from_json(json_config)
        self._model.load_weights(str(self._fname_weights))
        self.compile()




    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        if self.model_exists():
            logger.info(f"{self._fname} exists. skip fitting")
            return

        logger.info(f"start fitting")
        self.history = self._model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
                        sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size,
                        workers, use_multiprocessing)
        logger.info(f"finish fitting")
        self.save()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
             max_queue_size=10, workers=1, use_multiprocessing=False):
        return self._model.evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks,
             max_queue_size, workers, use_multiprocessing)

    def get_logit(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        return self._model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size,
                                   workers, use_multiprocessing)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        logit = self.get_logit(x, batch_size, verbose, steps, callbacks, max_queue_size,
                                   workers, use_multiprocessing)
        return tf.nn.softmax(logit)



