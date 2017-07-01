import tensorflow as tf

import sys
sys.path.append('../')

from sanity import *


class MemoryNet(object):

    def __init__(self, hdim, num_hops, vocab_size, lr=0.01):

        # initializer
        init = tf.random_normal_initializer(0, 0.1)

        # build placeholders
        questions = tf.placeholder(tf.int32, shape=[None, None], name='questions' )
        stories = tf.placeholder(tf.int32, shape=[None, None, None], name='stories' )
        answers = tf.placeholder(tf.int32, shape=[None, ], name='answers' )

        # embedding
        A = tf.get_variable('A', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)
        B = tf.get_variable('B', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)
        C = tf.get_variable('C', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)

        # embed questions
        u0 = tf.nn.embedding_lookup(B, questions)
        u0 = tf.reduce_sum(u0, axis=1)
        u = [u0] # accumulate question emb

        # more variables
        H = tf.get_variable('H', dtype=tf.float32, shape=[hdim, hdim],
                initializer=init)
        TA = tf.get_variable('TA', dtype=tf.float32, shape=[hdim],
                initializer=init)

        # embed stories
        m = tf.nn.embedding_lookup(A, stories)
        m = tf.reduce_sum(m, axis=2) + TA
        c = tf.nn.embedding_lookup(C, stories)
        c = tf.reduce_sum(c, axis=2)

        # memory loop
        for i in range(num_hops):
            p = tf.reduce_sum(m*tf.expand_dims(u[-1], axis=1), axis=-1)
            o = tf.reduce_sum(tf.expand_dims(p, axis=-1)*c, axis=1)
            u_k = tf.matmul(u[-1], H) + o
            u.append(u_k)

        # answer selection
        W = tf.get_variable('W', dtype=tf.float32, shape=[hdim, vocab_size],
                           initializer=init)
        self.logits = tf.matmul(u[-1], W)

        # optimization
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=answers
            )
        # attach loss to instance
        self.loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        # attach train op to instance
        self.train_op = optimizer.minimize(self.loss)


if __name__ == '__main__':

    memnet = MemoryNet(num_hops=3, hdim=150, vocab_size=1000, lr=0.01)

    print(sanity([memnet.loss, memnet.train_op]))
