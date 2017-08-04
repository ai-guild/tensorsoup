import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.goldilocks import MemoryNet
from train.trainer import Trainer
from tasks.cnn.data import DataSource
import tasks.cnn.proc as proc

from visualizer import Visualizer


if __name__ == '__main__':

    # optimal hyperparamters
    batch_size = 150
    window_size = 5

    datasrc = DataSource(datadir='../../../datasets/cnnqa/questions/', 
            window_size=window_size,
            batch_size=batch_size)

    # get vocab size from data source
    vocab_size = datasrc.metadata['vocab_size']
    memsize = datasrc.metadata['memory_size']
    sentence_size = datasrc.metadata['qlen']
    num_candidates = datasrc.metadata['max_candidates']

    # instantiate model
    model = MemoryNet(hdim=300, num_hops=1, memsize=memsize, 
                      window_size= window_size,
                      sentence_size=sentence_size, 
                      vocab_size=vocab_size,
                      num_candidates=num_candidates,
                      lr1 = 0.001, lr2=0.001)

    # setup visualizer
    #  by default, writes to ./log/
    vis = Visualizer()
    vis.attach_scalars(model)
    #vis.attach_params() # histograms of trainable variables

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
        trainer.fit(epochs=600, mode=Trainer.TRAIN, 
                verbose=True, visualizer=vis, 
                eval_interval=1,
                early_stop=False)
        '''
        print('****************************************************************** PRETRAINING OVER ')
        for task_id in reversed(range(21)):
            datasrc.task_id = task_id
            loss, acc = trainer.evaluate()
            print('evaluation loss for task_id = {}\t\tloss = {}\t\t accuracy = {}'.format(task_id, loss, acc))
        
        trainer.fit(epochs=600, mode=Trainer.TRAIN, verbose=False, visualizer=vis)
        print('****************************************************************** TRAINING OVER ')
        for task_id in reversed(range(21)):
            datasrc.task_id = task_id
            loss, acc = trainer.evaluate()
            print('evaluation loss for task_id = {}\t\tloss = {}\t\t accuracy = {}'.format(task_id, loss, acc))
        '''
