# Partially extrapolated from https://github.com/chawins/DART

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import call_model

# from parameters import *
# import utils
tf.compat.v1.disable_eager_execution()

def gradient_fn(model):

    y_true = K.placeholder(shape=(43, ))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=model.output)
    grad = K.gradients(loss, model.input)

    return K.function([model.input, y_true, K.learning_phase()], grad)


def fg(model, x, y, mask, target):

    x_adv = np.zeros(x.shape, dtype=np.float32)
    grad_fn = gradient_fn(model)

    for i, x_in in enumerate(x):
        call_model.printProgressBar(i+50, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)

        if target == True:
            grad = -1 * grad_fn([x_in.reshape(1,32,32,3), y[i], 0])[0][0]
        else:
            grad = grad_fn([x_in.reshape(1,32,32,3), y[i], 0])[0][0]


        mask_rep = np.repeat(mask[i, :, :, np.newaxis], 3, axis=2)
        grad *= mask_rep

        try:
            grad /= np.linalg.norm(grad)
        except ZeroDivisionError:
            raise

        x_adv[i] = x_in + grad * 3.5

    x_adv = np.clip(x_adv, 0, 1)

    return x_adv
