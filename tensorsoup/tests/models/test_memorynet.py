import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet import *
from sanity import *

class MemoryNetTest(tf.test.TestCase):


    def test_memorynet(self):

        with self.test_session():

            memnet = MemoryNet(num_hops=3, hdim=150, vocab_size=1000)
            self.assertTrue(sanity([memnet.loss, memnet.logits, memnet.train_op]))


if __name__ == '__main__':
    tf.test.main()
