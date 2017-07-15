import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.model import MemoryNet
from train.trainer import Trainer
from tasks.babi.data import DataSource


if __name__ == '__main__':

    batch_size = 32

    datasrc = DataSource(datadir='../../../datasets/babi/en/', task_id=1,
            batch_size=batch_size)

    # get vocab size from data source
    vocab_size = datasrc.metadata['vocab_size']
    memsize = datasrc.metadata['memory_size']
    sentence_size = datasrc.metadata['sentence_size']

    # instantiate model
    model = MemoryNet(hdim=20, num_hops=1, memsize=memsize, 
                      sentence_size=sentence_size, vocab_size=vocab_size,
                      lr = 0.01)

    # gpu config
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size)

        # fit model
        trainer.fit(epochs=500, verbose=False)
