import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.memn2n import MemoryNet
from train.trainer import Trainer
from tasks.babi.proc import gather
from datafeed import DataFeed

from visualizer import Visualizer


if __name__ == '__main__':

    batch_size = 128

    # get task 1
    task = 1
    data, metadata = gather('1k', task)

    # gather info from metadata
    num_candidates = metadata['candidates']['vocab_size']
    vocab_size = metadata['vocab_size']
    memsize = metadata['clen']
    sentence_size = metadata['slen']

    # build data format
    dformat = [ 'contexts', 'questions', 'answers' ]

    # create feeds
    trainfeed = DataFeed(dformat, data=data['train'])
    testfeed  = DataFeed(dformat, data=data['test' ])

    # instantiate model
    model = MemoryNet(hdim=20, num_hops=3, memsize=memsize, 
                      sentence_size=sentence_size, 
                      vocab_size=vocab_size,
                      num_candidates = num_candidates,
                      lr = 0.001)

    with tf.Session() as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # create trainer
        trainer = Trainer(sess, model, trainfeed, testfeed,
                batch_size = batch_size)

        print('\n:: [1/2] Pretraining')
        # pretrain
        trainer.fit(epochs=100000, eval_interval=100, 
                mode=Trainer.PRETRAIN, verbose=False)
        
        print('\n:: [1/2] Training')
        # train
        trainer.fit(epochs=10000, eval_interval=100, 
                mode=Trainer.TRAIN, verbose=False)
