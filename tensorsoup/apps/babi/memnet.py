import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.model import MemoryNet
from train.trainer import Trainer
from tasks.babi.data import DataSource

from visualizer import Visualizer


if __name__ == '__main__':

    batch_size = 128

    datasrc = DataSource(datadir='../../../datasets/babi/en-10k/', task_id=0,
            batch_size=batch_size)

    # get vocab size from data source
    vocab_size = datasrc.metadata['vocab_size']
    memsize = datasrc.metadata['memory_size']
    sentence_size = datasrc.metadata['sentence_size']

    # instantiate model
    model = MemoryNet(hdim=50, num_hops=3, memsize=memsize, 
                      sentence_size=sentence_size, vocab_size=vocab_size,
                      lr = 0.001)

    # setup visualizer
    #  by default, writes to ./log/
    vis = Visualizer()
    vis.attach_scalars(model)
    vis.attach_params() # histograms of trainable variables

    # gpu config
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # add graph to visualizer
        vis.attach_graph(sess.graph)

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size)

        # fit model
        trainer.fit(epochs=600, mode=Trainer.PRETRAIN, verbose=True, visualizer=vis)
        print('****************************************************************** PRETRAINING OVER ')
        trainer.fit(epochs=600, mode=Trainer.TRAIN, verbose=False, visualizer=vis)
        

