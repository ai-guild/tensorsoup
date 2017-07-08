import tensorflow as tf

import sys
sys.path.append('../')

from sanity import *


class MemoryNet(object):

    def __init__(self, hdim, num_hops, vocab_size, lr=0.01):

        # reset graph
        tf.reset_default_graph()

        # initializer
        init = tf.random_normal_initializer(0, 0.1)

        # build placeholders
        self.questions = tf.placeholder(tf.int32, shape=[None, None], name='questions' )
        self.stories = tf.placeholder(tf.int32, shape=[None, None, None], name='stories' )
        self.answers = tf.placeholder(tf.int32, shape=[None, ], name='answers' )

        # embedding
        A = tf.get_variable('A', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)
        B = tf.get_variable('B', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)
        C = tf.get_variable('C', shape=[vocab_size, hdim], dtype=tf.float32, 
                           initializer=init)

        # embed questions
        u0 = tf.nn.embedding_lookup(B, self.questions)
        u0 = tf.reduce_sum(u0, axis=1)
        u = [u0] # accumulate question emb

        # more variables
        H = tf.get_variable('H', dtype=tf.float32, shape=[hdim, hdim],
                initializer=init)
        TA = tf.get_variable('TA', dtype=tf.float32, shape=[hdim],
                initializer=init)

        # embed stories
        m = tf.nn.embedding_lookup(A, self.stories)
        m = tf.reduce_sum(m, axis=2) + TA
        c = tf.nn.embedding_lookup(C, self.stories)
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
            labels=self.answers
            )
        # attach loss to instance
        self.loss = tf.reduce_mean(cross_entropy)

        # evaluation
        probs = tf.nn.softmax(self.logits)
        correct_labels = tf.equal(tf.cast(self.answers, tf.int64), tf.argmax(probs, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))

        #optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # attach train op to instance
        self.train_op = optimizer.minimize(self.loss)

        # placeholders 
        self._placholders()

    # expose placeholders as ordered list
    def _placholders(self):
        self.placeholders = [ self.stories, self.questions, self.answers ]


if __name__ == '__main__':

    memnet = MemoryNet(num_hops=3, hdim=150, vocab_size=1000, lr=0.001)

    print(sanity([memnet.loss, memnet.train_op]))
