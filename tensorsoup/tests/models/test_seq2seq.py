import tensorflow as tf

import sys
sys.path.append('../../')

from recurrence import *
from models.seq2seq import *
from sanity import *

class Seq2seqTest(tf.test.TestCase):


    def test_naive_seq2seq(self):

        with self.test_session():

            d, vocab_size = 10, 100
            inputs = tf.placeholder(tf.float32, [None, None, d])
            targets = tf.placeholder(tf.int64, [None, None])
            emb = tf.get_variable('emb', [vocab_size, d], tf.float32)

            ecell = rcell('gru', d, dropout=0.3)
            dcell = rcell('gru', d, dropout=0.3)

            with tf.variable_scope('seq2seq') as scope:
                decoder_outputs_train = naive_seq2seq(inputs, targets, emb, ecell, dcell, 
                        training=True, feed_previous=False)

                scope.reuse_variables()

                decoder_outputs_test = naive_seq2seq(inputs, targets, emb, ecell, dcell, 
                        training=False, feed_previous=True)

            self.assertTrue(sanity([decoder_outputs_train, decoder_outputs_test]))


    def test_attentive_seq2seq(self):

        with self.test_session():

            d, vocab_size = 10, 100
            inputs = tf.placeholder(tf.float32, [None, None, d])
            targets = tf.placeholder(tf.int64, [None, None])
            emb = tf.get_variable('emb', [vocab_size, d], tf.float32)

            ecell = rcell('gru', d, dropout=0.3)

            with tf.variable_scope('attentive_seq2seq') as scope:
                decoder_outputs_train = attentive_seq2seq(inputs, targets, emb, ecell,
                        training=True, feed_previous=False)

                scope.reuse_variables()

                decoder_outputs_test = attentive_seq2seq(inputs, targets, emb, ecell,
                        training=False, feed_previous=True)

            self.assertTrue(sanity([decoder_outputs_train, decoder_outputs_test]))



if __name__ == '__main__':
    tf.test.main()
