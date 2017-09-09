import tensorflow as tf

import sys
sys.path.append('../../')

from models.asreader_graph import ASReaderGraph
from tasks.cbt.proc import gather
from tasks.cbt.proc import FIELDS

from train.trainer import Trainer
from models.model import Model
from datafeed import DataFeed
from graph import *


if __name__ == '__main__':

    # get data for task
    data, lookup, metadata = gather()

    # build data format
    dformat = FIELDS + ['cmask']

    # create feeds
    trainfeed = DataFeed(dformat, data=data['valid'])
    testfeed  = DataFeed(dformat, data=data['test' ])

    # training params
    batch_size = 32

    # instantiate model
    model = Model(ASReaderGraph, dformat=dformat, n=1,
            optimizer=tf.train.AdamOptimizer,
            lr=0.001,
            vocab_size=metadata['vocab_size'],
            max_candidates= metadata['max_candidates'],
            demb=384, dhdim=384,
            num_layers=1)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        # init session
        sess.run(tf.global_variables_initializer())

        # create trainer
        trainer = Trainer(sess, model, trainfeed, testfeed,
                batch_size = batch_size)

        # train
        acc = trainer.fit(epochs=1000000, 
                    eval_interval=1,
                    mode=Trainer.TRAIN, 
                    verbose=True,
                    lr=0.001)

        print(':: \tAccuracy after training: ', acc)
