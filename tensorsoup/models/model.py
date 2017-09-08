import sys
sys.path.append('../')

import tensorflow as tf

from models.asreader_graph import ASReaderGraph
from sanity import sanity


class Model(object):
    def __init__(self, _graph, dformat, n=1, 
            optimizer=tf.train.AdamOptimizer,
            *args, **kwargs):

        # clear global tf.graph
        tf.reset_default_graph()

        self.n = n # num of copies of graph
        self._graph = _graph # handle to custom Graph class
        self.optimizer = optimizer
        self.dformat = dformat
        
        self.make(*args, **kwargs)

        #g = self._graph(*args, **kwargs)

        # make model (n)
        #  [1] make_1
        #  [2] make_parallel
        #self.make(*args, **kwargs)

    def make(self, *args, **kwargs):
        # check n
        if self.n==1:
            g = self.make_1(*args, **kwargs)

            # get loss
            self.loss = g.loss
            self.accuracy = g.accuracy
            self.prob = g.prob

            # get placeholders
            self.placeholders = [ g.placeholders[k] for k in self.dformat ]

        else:
            #g = self.make_parallel(*args, **kwargs)
            pass

        # run optimize to get train_op
        self.optimize()

    def make_1(self, *args, **kwargs):
        # create 1 instance of graph
        return self._graph(*args, **kwargs)

    def optimize(self, clip_norm=10.):
        with tf.name_scope('optimization'):
            # create optimizer
            optimizer = self.optimizer()

            # gradient clipping
            gvs = optimizer.compute_gradients(self.loss)
            clipped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(clipped_gvs)



if __name__ == '__main__':
    vocab_size = 100
    max_candidates = 10
    demb = 32
    dhdim = 32
    num_layers = 1

    dformat = ['context', 'query', 'answer', 'candidates', 'cmask']

    model_ = Model(ASReaderGraph, dformat=dformat, n=1, 
            optimizer=tf.train.AdamOptimizer, 
            vocab_size=100, 
            max_candidates=10, 
            demb=32,
            dhdim=32, 
            num_layers=1)

    results = sanity([model_.prob, model_.loss, model_.accuracy], 
            fetch_data=True)
    print(results[0])
