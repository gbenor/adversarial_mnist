from pathlib import Path
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model

from attacks import FastGradientSignMethod, TargetedGradientSignMethod, BasicIterativeMethod
from consts import CONV_MINST_FNAME, LEARNING_RATE, LR_DROP, BATCH_SIZE, MAXEPOCHES, num_classes
from conv_minst import ConvMinst
from distillation import Distillation
from utils import load_data, lr_scheduler, TestAttack





def main():
    train_images, train_labels, test_images, test_labels = load_data()
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    dist = Distillation("dist", 30)
    dist.fit(train_images, train_labels,
                   batch_size=BATCH_SIZE,
                   epochs=MAXEPOCHES,
                   verbose=1,
                   validation_data=(test_images, test_labels),
                   callbacks=[reduce_lr])
    dist.add_softmax_layer()
    # exit(3)
    #
    # Train simple MINST conv network
    # conv_minst = ConvMinst(CONV_MINST_FNAME)
    # conv_minst.fit(train_images, train_labels,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=MAXEPOCHES,
    #                     verbose=1,
    #                     validation_data=(test_images, test_labels),
    #                     callbacks=[reduce_lr])
    # conv_minst.add_softmax_layer()

    attacks = {
        "NoAttack" : {"targeted": False,
                            "func": FastGradientSignMethod,
                            "epsilon": 10e-12},
                "FastGradientSignMethod" : {"targeted": False,
                                           "func": FastGradientSignMethod,
                                           "epsilon": 0.3},
               "TargetedGradientSignMethod" : {"targeted": True,
                                           "func": TargetedGradientSignMethod,
                                            "epsilon": 0.3},
               "Targeted_BIM" :  {"targeted": True,
                                  "func": BasicIterativeMethod,
                                  "epsilon" : 4.0},
               "UnTargeted_BIM": {"targeted": False,
                                  "func": BasicIterativeMethod,
                                  "epsilon" : 4.0}
                          }


    # model = conv_minst.model()
    model = dist.model()
    for attack_name, attack in attacks.items():
        target = (np.argmax(test_labels, axis=1) + np.random.randint(1, num_classes,
                                                                     size=(test_labels.shape[0]))) % num_classes
        target = keras.utils.to_categorical(target, num_classes)
        images = tf.Variable(test_images)

        attack_labels = tf.Variable(target) if attack["targeted"] else tf.Variable(test_labels)
        adv_images = attack["func"](model, images, attack_labels, epsilon=attack["epsilon"],
                                    iter_eps = 0.05, targeted=attack["targeted"])

        name = f"{model.name()}_{attack_name}"
        TestAttack(model, adv_images, test_images, test_labels, target, targeted=attack["targeted"], prefix=name)





if __name__ == '__main__':
    main()

