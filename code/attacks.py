from abc import ABC, abstractmethod
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models  import load_model

from utils import categorical_crossentropy_with_temp

def gradient_calc(model, images, labels):
    with tf.GradientTape() as tape:
        tape.watch(images)
        # adv_loss = categorical_crossentropy_with_temp(labels, model(images),1)
        adv_loss = keras.losses.categorical_crossentropy(labels, model(images))
    return tape.gradient(adv_loss, images)


''' Fast Gradient Sign Method implementation - perturb all input features by an epsilon sized step in
    the direction of loss gradient
'''
def FastGradientSignMethod(model, images, labels, epsilon, iter_eps=None, targeted=None):
    adv_grads = gradient_calc(model, images, labels)
    adv_out = images + epsilon * tf.sign(adv_grads)
    return adv_out

''' Targeted Gradient Sign Method implementation - A targeted variant of the FGSM attack
    here we minimize the loss with respect to the target class, as opposed to maximizing the loss with respect
    to the source class
'''
def TargetedGradientSignMethod(model, images, target, epsilon, iter_eps=None, targeted=None):
    adv_grads = gradient_calc(model, images, target)
    adv_out = images - epsilon * tf.sign(adv_grads)
    return adv_out


def BasicIterativeMethod(model, images, labels, epsilon, iter_eps, iterations=10, targeted=False):
    adv_out = tf.identity(images)
    targeted_tensor = tf.constant(targeted)

    def cond(images, adv_out):
        return True

    def perturb(images, adv_out):
        # Perturb with FGSM or TGSM
        def tgsm_base():
            return TargetedGradientSignMethod(model, adv_out, labels, epsilon=iter_eps)

        def fgsm_base():
            return FastGradientSignMethod(model, adv_out, labels, epsilon=iter_eps)

        adv_out = tf.cond(targeted_tensor, tgsm_base, fgsm_base)

        # Project the perturbation to the epsilon ball
        perturbation = adv_out - images
        norm = tf.reduce_sum(tf.square(perturbation), axis=(1, 2, 3), keepdims=True)
        norm = tf.sqrt(tf.maximum(10e-12, norm))
        factor = tf.minimum(1.0, tf.divide(epsilon, norm))
        adv_out = tf.clip_by_value(images + perturbation * factor, 0.0, 1.0)

        return images, adv_out

    _, adv_out = tf.while_loop(cond, perturb, (images, adv_out), back_prop=True, maximum_iterations=iterations)

    return adv_out

