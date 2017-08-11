import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected



def batch_norm_relu(inputs, output_shape, phase = True, scope = None, activation = True):
    with tf.variable_scope(scope):
        h1 = fully_connected(inputs, output_shape, activation_fn= None, scope ="dense")
        h2 = batch_norm(h1, decay = 0.95, center = True, scale = True,
                        is_training= phase, scope = 'bn', updates_collections=None)
        if activation:
            out = tf.nn.relu(h2, 'relu')
        else:
            out = h2

    return out
