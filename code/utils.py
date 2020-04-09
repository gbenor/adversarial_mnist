import os
from pathlib import Path

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.backend import categorical_crossentropy


from adv_logger import logger
from consts import LEARNING_RATE, LR_DROP


def lr_scheduler(epoch):
    return LEARNING_RATE * (0.5 ** (epoch // LR_DROP))


''' Normalize input to the range of [0..1]
    Apart from assisting in the convergance of the training process, this
    will also make our lives easier during the adversarial attack process
'''
def normalize(x_train, x_test):
    x_train -= x_train.min()
    x_train /= x_train.max()
    x_test -= x_test.min()
    x_test /= x_test.max()

    return x_train, x_test



# Load and prepare the datasets for training
def load_data():
    num_classes = 10

    img_rows, img_cols, img_colors = 28, 28, 1
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    train_images, test_images = normalize(train_images, test_images)

    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    return train_images, train_labels, test_images, test_labels

def get_target_label(test_labels, num_classes):
    target = (np.argmax(test_labels, axis=1) + np.random.randint(1, num_classes,
                                                                 size=(test_labels.shape[0]))) % num_classes
    target = keras.utils.to_categorical(target, num_classes)
    return target


''' A simple utility funcion for evaluating the success of an attack
'''
def TestAttack(model, adv_images, orig_images, true_labels,
               target=None, target_labels=None, targeted=False, prefix=None):
    result = {}
    def attack_info(k, v):
        logger.info(f"{prefix}: {k}: {v:.2f}")
        result[k] = v


    score = model.evaluate(adv_images, true_labels, verbose=0)
    attack_info('Test loss', score[0])
    attack_info('Successfully moved out of source class', (1 - score[1]))

    if targeted:
        score = model.evaluate(adv_images, target, verbose=0)
        attack_info('Test loss', score[0])
        attack_info('Successfully perturbed to target class', score[1])

    dist = np.mean(np.sqrt(np.mean(np.square(adv_images - orig_images), axis=(1, 2, 3))))
    attack_info('Mean perturbation distance', dist)

    fig = plt.figure(figsize=(16, 10))
    for row, sample_id in enumerate(range(10,13)):
        figs = {"original" : orig_images[sample_id].reshape(28, 28),
                "adversarial": adv_images[sample_id].numpy().reshape(28, 28),
                "perturbation": (adv_images[sample_id].numpy()-orig_images[sample_id]).reshape(28, 28)}

        perturbation_norm = LA.norm(figs["perturbation"], ord="fro")

        for col, (title, img) in enumerate(figs.items()):
            if title=="perturbation":
                title = f"perturbation={perturbation_norm:.3f}"
            ax = fig.add_subplot(3, 3, row*3 + col + 1, title=title)
            plt.imshow(img, cmap='gray')

    plt.savefig(f"{prefix}.pdf", format="pdf", bbox_inches='tight')
    return result



def categorical_crossentropy_with_temp(correct, predicted, temp=1):
    return categorical_crossentropy(target=correct, output=predicted / temp, from_logits=True)