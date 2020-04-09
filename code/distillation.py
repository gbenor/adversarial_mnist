import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model
from tensorflow.keras.backend import categorical_crossentropy

from abc_model import AbcModel
from adv_logger import logger
from consts import num_classes
from conv_minst import ConvMinst


class Distillation (AbcModel):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        teacher = ConvMinst("teacher.h5", self.temp())
        student = ConvMinst("student.h5", self.temp())

        logger.info(f"start fitting the teacher")
        self.teacher_history = teacher.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
                        sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size,
                        workers, use_multiprocessing)

        logger.info("evaluate the labels at temperature t")
        teacher_logit = teacher.get_logit(x)
        teacher_predicted = tf.nn.softmax(teacher_logit / self._temp)

        logger.info("train the student model at temperature t")
        self.student_history = student.fit(x, teacher_predicted, batch_size, epochs, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight,
                                           sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                                           validation_freq, max_queue_size,
                                           workers, use_multiprocessing)

        self._model = student.get_model()

    def build(self):
        pass

    def compile(self):
        pass
