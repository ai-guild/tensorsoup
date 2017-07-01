import tensorflow as tf

import sys
sys.path.append('../')

from recurrence import *
from sanity import *


class RecurrenceTest(tf.test.TestCase):

    def test_rcell(self):

        with self.test_session():
            d = 90
            inputs = tf.placeholder(tf.float32, [None, None, d])
            inputs_tm = tf.transpose(inputs, [1,0,2], name='inputs_tm')

            cell_ = rcell('gru', d, num_layers=3, dropout=0.3)
            zero_state = cell_.zero_state(tf.shape(inputs)[0], tf.float32)

            self.assertTrue(sanity(cell_(inputs_tm[0], zero_state)))

    def test_uni_net_dynamic(self):
        with self.test_session():
            d = 90
            inputs = tf.placeholder(tf.float32, [None, None, d])
            inputs_tm = tf.transpose(inputs, [1,0,2], name='inputs_tm')
            cell_ = rcell('gru', d, dropout=0.3)
            self.assertTrue(sanity(uni_net_dynamic(cell_, inputs)))

    def test_bi_net_dynamic(self):
        with self.test_session():
            d = 90
            inputs = tf.placeholder(tf.float32, [None, None, d])
            inputs_tm = tf.transpose(inputs, [1,0,2], name='inputs_tm')
            cell_f = rcell('lstm', d, dropout=0.3)
            cell_b = rcell('lstm', d, dropout=0.3)
            self.assertTrue(sanity(bi_net_dynamic(cell_f, cell_f, inputs)))

if __name__ == '__main__':
    tf.test.main()
