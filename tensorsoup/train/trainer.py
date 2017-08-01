from tqdm import tqdm
import tensorflow as tf

import numpy as np


class Trainer(object):

    PRETRAIN = 0
    TRAIN = 1
    TEST = 2

    def __init__(self, sess, model, datasrc, batch_size, rand=None):
        self.model = model
        self.datasrc = datasrc
        self.sess = sess
        self.rand = rand
        

    def evaluate(self, visualizer=None):
        datasrc = self.datasrc
        batch_size = datasrc.batch_size
        num_examples = datasrc.n['test']#getN('test')
        model = self.model

        # num copies
        n = model.n

        num_iterations = int(num_examples / (batch_size*n))

        build_feed = self.build_feed_dict if n == 1 else self.build_feed_dict_multi

        avg_loss, avg_acc = 0., 0.

        for i in tqdm(range(num_iterations)):
            bi = datasrc.next_batch(n, 'test')

            fetch_data = [model.loss, model.accuracy, model.train_op]

            if visualizer:
                fetch_data.append(visualizer.summary_op)

            feed_dict = self.extra_params(self.TEST, build_feed(model.placeholders, bi))
            results = self.sess.run( fetch_data,
                                     feed_dict = feed_dict)

            l, acc = results[:2]

            if visualizer:
                if i % visualizer.interval == 0:
                    visualizer.eval_log(results[-1], i)


            avg_loss += l
            avg_acc += acc

        log = 'Evaluation - loss : {}; accuracy : {}'.format(avg_loss/(num_iterations),
                            avg_acc/(num_iterations))
        tqdm.write(log)
        return avg_loss/num_iterations, avg_acc/(num_iterations)

    def fit(self, epochs, eval_interval=10, mode=1, 
            verbose=True, visualizer=None,
            early_stop=True):

        def tq(x):
            return tqdm(x) if verbose else x

        model = self.model
        datasrc = self.datasrc
        batch_size = self.datasrc.batch_size
        sess = self.sess

        # num copies of model
        n = model.n
        
        # get count of data
        num_examples = datasrc.n['train']
        if self.rand:
            num_examples = num_examples * datasrc.random_x

        num_iterations = int(num_examples/(batch_size*n))

        build_feed = self.build_feed_dict if n == 1 else self.build_feed_dict_multi
        loss_trend = []
        for i in range(epochs):
            avg_loss, avg_acc = 0., 0.
            for j in tq(range(num_iterations)):

                next_batch = datasrc.next_random_batch if self.rand else datasrc.next_batch

                bj = next_batch(n, 'train')

                # fetch items
                fetch_data = [model.loss, model.accuracy, model.train_op]

                if visualizer:
                    fetch_data.append(visualizer.summary_op)
                    
                feed_dict = self.extra_params(mode, build_feed(model.placeholders, bj))

                results = sess.run( fetch_data, 
                                    feed_dict = feed_dict)
                
                l, acc = results[:2]

                if visualizer:
                    if i % visualizer.interval == 0:
                        visualizer.train_log(results[-1], j)

                # accumulate loss, accuracy
                avg_loss += l
                avg_acc += acc

            if verbose:
                log = '[{}] loss : {}; accuracy : {}'.format(i,
                        avg_loss/(num_iterations), avg_acc/(num_iterations))
                tqdm.write(log)
            
            # eval
            if i and i%eval_interval == 0:
                eloss = self.evaluate(visualizer)
                loss_trend.append(eloss)

                if early_stop:
                    if early_stopping(loss_trend):
                        tqdm.write('stopping from early stopping')
                        return

    def build_feed_dict_multi(self, ll1, ll2):
        feed_dict = {}
        # we have a tuple
        #  [1] list of list of placeholders
        #  [2] list of list of data in batches
        for l1, l2 in zip(ll1, ll2):
            for i,j in zip(l1,l2):
                feed_dict[i] = j

        return feed_dict

    def build_feed_dict(self, l1, l2):
        return { i:j for i,j in zip(l1,l2)}

    def extra_params(self, mode, feed_dict):
        feed_dict[self.model.mode] = mode
        return feed_dict


def early_stopping(loss_trend):
    if len(loss_trend) < 4:
        return False
    
    for p, n in zip(loss_trend[-4:], loss_trend[-3:]):
        if p > n:
            return False
    return True
