import tensorflow as tf


def mask_emb(emb):
    shapes = tf.unstack(tf.shape(emb))
    shapes_0 = [1] + shapes[1:]
    shapes = [shapes[0]-1] + shapes[1:]
    return tf.concat([tf.zeros(shapes_0, dtype=tf.float32),
        tf.ones(shapes, dtype=tf.float32)], axis=0) * emb
