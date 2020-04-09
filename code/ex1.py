from pathlib import Path
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model

from attacks import FastGradientSignMethod, TargetedGradientSignMethod, BasicIterativeMethod
from consts import LEARNING_RATE, LR_DROP, BATCH_SIZE, MAXEPOCHES, num_classes, RESULTS_DF_CSV_NAME
from conv_minst import ConvMinst
from distillation import Distillation
from utils import load_data, lr_scheduler, TestAttack, get_target_label


def main():
    #load data
    train_images, train_labels, test_images, test_labels = load_data()
    target_labels = get_target_label(test_labels, num_classes)

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    models = [{"name" : "conv_minst",
              "class" : ConvMinst,
              "temp" : [1]},

             {"name": "distillation",
              "class": Distillation,
              "temp": range(1, 40, 5)}
    ]

    attacks = {
        "NoAttack": {"targeted": False,
                     "func": FastGradientSignMethod,
                     "epsilon": 10e-12},
        "FastGradientSignMethod": {"targeted": False,
                                   "func": FastGradientSignMethod,
                                   "epsilon": 0.3},
        "TargetedGradientSignMethod": {"targeted": True,
                                       "func": TargetedGradientSignMethod,
                                       "epsilon": 0.3},
        "Targeted_BIM": {"targeted": True,
                         "func": BasicIterativeMethod,
                         "epsilon": 4.0},
        "UnTargeted_BIM": {"targeted": False,
                           "func": BasicIterativeMethod,
                           "epsilon": 4.0}
    }

    index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                          names=["architecture", "temp", "attack"])
    res_df = pd.DataFrame(index=index)
    for model_dict in models:
        for temp in model_dict["temp"]:
            clf = model_dict["class"](model_dict["name"], temp)
            clf.fit(train_images, train_labels,
                           batch_size=BATCH_SIZE,
                           epochs=MAXEPOCHES,
                           verbose=1,
                           validation_data=(test_images, test_labels),
                           callbacks=[reduce_lr])
            clf.add_softmax_layer()

            for attack_name, attack in attacks.items():
                images = tf.Variable(test_images)
                attack_labels = tf.Variable(target_labels) if attack["targeted"] else tf.Variable(test_labels)
                adv_images = attack["func"](clf.get_model(), images, attack_labels, epsilon=attack["epsilon"],
                                            iter_eps = 0.05, targeted=attack["targeted"])
                model_name = model_dict["name"]
                cnfg_name = f"{model_name}_{temp}_{attack_name}"
                test_result = TestAttack(clf.get_model(), adv_images, test_images, test_labels, target_labels, targeted=attack["targeted"],
                           prefix=cnfg_name)
                for k, v in test_result.items():
                    res_df.loc[(model_dict["name"], temp,  attack_name), k] = v
                print (res_df)
            res_df.to_csv(RESULTS_DF_CSV_NAME)






if __name__ == '__main__':
    main()

