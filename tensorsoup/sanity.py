import tensorflow as tf
import numpy as np


def sanity(t, g=None, default_shape=[8,20,100], int_limit = [0, 1000], fetch_data=False):
    g = tf.get_default_graph() if not g else g
    placeholder_ops = [ op for op in g.get_operations() if op.type=='Placeholder' ]
    placeholders = [ g.get_tensor_by_name(op.name + ':0') for op in placeholder_ops ]
    feed_dict = {}
    for placeholder in placeholders:
        dtype= placeholder.dtype
        
        shape = placeholder.shape.as_list()
        for i,shp in enumerate(shape):
            if not shp:
                shape[i] = default_shape[i]
                
        if dtype == tf.float32:
            feed_dict[placeholder] = np.random.uniform(-0.9, 0.9, shape)
        else:
            feed_dict[placeholder] = np.random.randint(int_limit[0], int_limit[1],
                    shape, dtype=np.int64)
            
    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(t, feed_dict) if fetch_data else True
    except Exception as e:
        return 'Something went wrong\n\n' + str(e)
