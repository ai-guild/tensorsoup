import tensorflow as tf


'''
    set trainable variables

'''
def set_op(values):
    return [ tf.assign(var, val) for var, val in
            zip(tf.trainable_variables(), values) ]

'''
    get trainable variables

'''
def get_op(sess):
    return sess.run(tf.trainable_variables())
