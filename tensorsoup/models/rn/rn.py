import tensorflow as tf
import numpy as np

import sys
sys.path.append('../../')

from sanity import *
from recurrence import *

from collections import OrderedDict
import itertools



def seqlen(seq):
    return tf.reduce_sum(tf.cast(seq > 0, tf.int32), 
            axis=-1)


class RelationNet(object):

    def __init__(self, clen, qlen, slen, vocab_size, 
            lstm_units=32,
            g_hdim = [256,256,256],
            f_hdim = [256,512, None],
            lr=0.025):

        # final output shape
        f_hdim[-1] = vocab_size

        # context size : num of sentences in context
        #  is hardcoded to "20" in the model
        clen_max = 20

        # reset graph
        tf.reset_default_graph()

        # initializer
        self.init = tf.random_uniform_initializer(-1., 1.)

        # num of copies of model (for multigpu training)
        self.n = 1

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)


        def inference():

            with tf.name_scope('input'):

                # placeholders
                queries = tf.placeholder(tf.int32, shape=[None, qlen], 
                        name='queries')
                context = tf.placeholder(tf.int32, shape=[None, clen, slen],
                        name='context')
                answers = tf.placeholder(tf.int32, shape=[None, ], 
                        name='answers')
                mode = tf.placeholder(tf.int32, shape=(), name='mode')

            # expose handle to placeholders
            placeholders = OrderedDict()
            placeholders['queries'] = queries
            placeholders['context'] = context
            placeholders['answers'] = answers

            # get last 20 sentences in context
            context_sliced = tf.slice(tf.reverse(context, axis=[1]), 
                        [0,0,0], [-1,clen_max,-1])

            # get batch size
            batch_size = tf.shape(queries)[0]

            with tf.name_scope('embedding'):
                qemb_mat = tf.get_variable('qemb_mat', shape=[vocab_size, lstm_units], dtype=tf.float32,
                                          initializer=self.init)
                cemb_mat = tf.get_variable('cemb_mat', shape=[vocab_size, lstm_units], dtype=tf.float32,
                                          initializer=self.init)

                qemb = tf.nn.embedding_lookup(qemb_mat, queries)
                cemb = tf.nn.embedding_lookup(cemb_mat, context_sliced)

            # question LSTM
            with tf.variable_scope('question_lstm'):
                q_rcell = rcell('lstm', lstm_units)
                _, q_final_state = tf.nn.dynamic_rnn(q_rcell, inputs=qemb, 
                                     sequence_length=seqlen(queries), 
                                     initial_state=q_rcell.zero_state(batch_size, 
                                                                  tf.float32))
                qo = tf.concat([q_final_state.c, q_final_state.h], 
                               axis=-1)

            with tf.variable_scope('context_lstm'):
                # cell for encoding context
                c_rcell = rcell('lstm', lstm_units)
                # list of context objects
                co = []
                # iterate through sentences in context
                cemb_reshaped = tf.transpose(cemb, [1,0,2,3])
                c_reshaped = tf.transpose(context_sliced, [1,0,2])
                #  [clen, batch_size, slen, lstm_units]
                for i in range(clen_max):
                    # get final state
                    context_i = cemb_reshaped[i]
                    _, final_state = tf.nn.dynamic_rnn(c_rcell, 
                                                       inputs=context_i,
                                                       sequence_length=seqlen(c_reshaped[i]),
                                                       initial_state=c_rcell.zero_state(batch_size,
                                                                                       tf.float32)
                                                      )
                    tf.get_variable_scope().reuse_variables()
                    co.append(tf.concat([final_state.c,
                                        final_state.h], axis=-1, 
                                        name='object_' + str(i)))

            # object pairs
            with tf.variable_scope('object_pairs'):
                rn_inputs = []
                for object_pair in list(itertools.combinations(co, 2)):
                    rn_input = tf.concat([object_pair[0], object_pair[1],
                                          qo], axis=-1)
                    rn_inputs.append(rn_input)
                # concat to tensor
                rn_inputs = tf.concat(rn_inputs, axis=0)

            # g(theta) 
            with tf.variable_scope('g_theta', reuse=None) as scope:
                g = rn_inputs
                for hdim in g_hdim:
                    g = tf.contrib.layers.fully_connected(g, hdim)
                g = tf.reshape(g, [-1,batch_size, hdim])

            # f(theta)
            with tf.variable_scope('f_phi', reuse=None) as scope:
                f = tf.reduce_sum(g, axis=0)
                for hdim in f_hdim:
                    f = tf.contrib.layers.fully_connected(f, hdim)

            # final representation -> logits
            logits = f
            
            # get prob
            probs = tf.nn.softmax(logits)
            # prediction
            prediction = tf.argmax(probs, axis=-1)

            with tf.name_scope('loss'):
                # optimization
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=answers
                    )
                # attach loss to instance
                loss = tf.reduce_mean(cross_entropy)

            with tf.name_scope('evaluation'):
                # evaluation
                correct_labels = tf.equal(tf.cast(answers, tf.int64), tf.argmax(probs, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))

            # optimization
            with tf.name_scope('optimization'):
                # gradient clipping
                gvs = optimizer.compute_gradients(loss)
                clipped_gvs = [(tf.clip_by_norm(grad, 40.), var) for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(clipped_gvs)


            self.logits = logits
            self.loss = loss
            self.accuracy = accuracy
            self.context = context
            self.queries = queries
            self.answers = answers
            self.mode = mode

            self.placeholders = [context, queries, answers]

        # execute inference and build graph
        inference()


if __name__ == '__main__':

    # instantiate RN
    rn = RelationNet(clen=50, qlen=12, slen=14,
            vocab_size = 120, lr=0.0002)

    # sanity check
    print(sanity(rn.logits, fetch_data=True))