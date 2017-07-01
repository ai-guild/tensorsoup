from init import *
import tensorflow as tf

#@init
def get_variables(n, shape, name='W'):
    return [tf.get_variable(name+str(i), dtype=tf.float32, shape=shape)#, initializer=def_init)
               for i in range(n)]
