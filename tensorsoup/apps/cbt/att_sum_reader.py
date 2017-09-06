import tensorflow as tf

import sys
sys.path.append('../../')

from models.asreader import ASReader
from tasks.cbt.proc import gather
from tasks.cbt.proc import FIELDS

from train.trainer import Trainer
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
    batch_size = 64

    # instantiate model
    model = ASReader( vocab_size=metadata['vocab_size'], 
            max_candidates= 10, # TODO : fix this ||KeyError: 'max_candidates'||
            demb=38, dhdim=38,
            num_layers=3)

    with tf.Session() as sess:

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




