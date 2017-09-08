import sys
sys.path.append('../')

import tensorflow as tf

from models.asreader_graph import ASReaderGraph
from multigpu import average_gradients

from sanity import sanity


class Model(object):
    def __init__(self, _graph, dformat, n=1, 
            optimizer=tf.train.AdamOptimizer,
            lr=0.001,
            GPUs = [0],
            *args, **kwargs):

        # clear global tf.graph
        tf.reset_default_graph()

        self.n = n # num of copies of graph
        self.GPUs = GPUs
        self._graph = _graph # handle to custom Graph class
        self.optimizer = optimizer(lr)
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
            self.make_1(*args, **kwargs)

        else:
            self.make_parallel(*args, **kwargs)

        # run optimize to get train_op
        self.optimize()

    def make_1(self, *args, **kwargs):
        # create 1 instance of graph
        g = self._graph(*args, **kwargs)
        # get loss
        self.loss = g.loss
        self.accuracy = g.accuracy
        self.prob = g.prob

        # get placeholders
        self.placeholders = [ g.placeholders[k] for k in self.dformat ]

        # compute gradients
        self.grads = self.optimizer.compute_gradients(g.loss)


    def make_parallel(self, *args, **kwargs):

        # keep track of placholder and gradients
        tower_grads, ph_list, losses, accuracies = [], [], [], []
        probs = []

        with tf.variable_scope(tf.get_variable_scope()):
            # iterate through list of GPUs
            for i,GPU_ in enumerate(self.GPUs):
                # for each GPU[i]
                with tf.device('/gpu:{}'.format(GPU_)):
                    # create n/m copies per GPU
                    for j in range(self.n//len(self.GPUs)):
                        # separate name scope for each copy
                        with tf.name_scope('gpu_{}_{}'.format(i,j)) as scope:
                            # run inference
                            #  get handles to placeholders, logits and probs
                            g = self._graph(*args, **kwargs)

                            # reuse trainable parameters
                            tf.get_variable_scope().reuse_variables()

                            # gather gradients
                            grads = self.optimizer.compute_gradients(g.loss)

                            # save grads for averaging later
                            tower_grads.append(grads)

                            # save the list of placholder handles
                            ph_list.append([g.placeholders[k] 
                                for k in self.dformat])

                            # save loss and accuracy
                            losses.append(g.loss)
                            accuracies.append(g.accuracy)
                            # and probabilities
                            probs.append(g.prob)
        # in CPU:0
        with tf.device('/cpu:0'):
            # average gradients in cpu
            self.grads = average_gradients(tower_grads)
            # mean loss, accuracy
            self.loss = tf.reduce_mean(losses)
            self.accuracy = tf.reduce_mean(accuracies)
            self.prob = tf.reduce_mean(tf.stack(probs), axis=0)
            # handle to placeholders list[list]
            self.placeholders = ph_list


    def optimize(self, clip_norm=10.):
        with tf.name_scope('optimization'):
            # gradient clipping
            clipped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) 
                    for grad, var in self.grads]
            self.train_op = self.optimizer.apply_gradients(clipped_gvs)



if __name__ == '__main__':
    vocab_size = 100
    max_candidates = 10
    demb = 32
    dhdim = 32
    num_layers = 1

    dformat = ['context', 'query', 'answer', 'candidates', 'cmask']

    model_ = Model(ASReaderGraph, dformat=dformat, n=2,
            optimizer=tf.train.AdamOptimizer, 
            vocab_size=100, 
            max_candidates=10, 
            demb=32,
            dhdim=32, 
            num_layers=1)

    results = sanity([model_.prob, model_.loss, model_.accuracy], 
            fetch_data=True)
    print(results[0])
