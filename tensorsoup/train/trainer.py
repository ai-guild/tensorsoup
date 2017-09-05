from tqdm import tqdm
import tensorflow as tf
import numpy as np

from graph import *


class Trainer(object):

    PRETRAIN = 0
    TRAIN = 1
    TEST = 2

    def __init__(self, sess, model,
            trainfeed=None, testfeed=None, # optional data feeds
            lr=0.001,  # learning rate
            batch_size=1):
 
        self.model = model
        self.trainfeed = trainfeed
        self.testfeed = testfeed
        self.sess = sess
        self.batch_size = batch_size
        self.lr = lr


    def evaluate(self, feed=None, 
            batch_size=None,
            visualizer=None):

        # convenience
        model = self.model

        # set feed
        feed = feed if feed else self.testfeed
        # set batch size
        batch_size = batch_size if batch_size else self.batch_size

        # get num of examples
        num_examples = feed.getN()
        # get num of iterations
        num_iterations = num_examples // batch_size

        # maintain avg loss, accuracy
        avg_loss, avg_acc = 0., 0.
        for i in range(num_iterations):
            # get next batch
            bi = feed.next_batch(batch_size)

            # fetch loss and accuracy from graph
            fetch_data = [model.loss, model.accuracy]

            # add summary op to fetch_data if necessary
            if visualizer:
                fetch_data.append(visualizer.summary_op)
            
            # build feed_dict
            feed_dict = self.extra_params(
                    self.build_feed_dict(model.placeholders, bi), # feed_dict
                    self.TEST)

            # execute graph in session
            results = self.sess.run( fetch_data,
                                     feed_dict = feed_dict)
            # get loss, accuracy
            l, acc = results[:2]

            # log visualization summary
            if visualizer:
                if i % visualizer.interval == 0:
                    visualizer.eval_log(results[-1], i)

            # maintain average loss, accuracy
            avg_loss += 10 if np.isnan(l) else l
            avg_acc += 0 if np.isnan(acc) else acc

        # print info
        log = 'Evaluation - loss : {}; accuracy : {}'.format(avg_loss/(num_iterations),
                            avg_acc/(num_iterations))
        tqdm.write(log)

        # return average loss and accuracy
        acc = avg_acc/num_iterations
        loss = avg_loss/num_iterations
        acc = acc if acc else 0
        loss = loss if loss else 100
        return loss, acc
    

    def fit(self, epochs, 
            eval_interval=0, mode=1, lr=None,
            batch_size=None, feed=None, verbose=True, 
            visualizer=None, early_stop_after=1):

        def tq(x):
            return tqdm(x) if verbose else x

        model = self.model
        sess = self.sess

        # set batch size
        batch_size = batch_size if batch_size else self.batch_size
        # set feed
        feed = feed if feed else self.trainfeed
        # set learning rate
        lr = lr if lr else self.lr

        # get num of iterations
        num_examples = feed.getN()
        num_iterations = num_examples//batch_size

        #build_feed = self.build_feed_dict if n == 1 else self.build_feed_dict_multi

        loss_trend, accuracies = [], []
        model_params = None
        for i in range(epochs):
            avg_loss = 0.
            for j in tq(range(num_iterations)):

                bj = feed.next_batch(batch_size)

                # fetch items
                fetch_data = [model.loss, model.train_op]

                if visualizer:
                    fetch_data.append(visualizer.summary_op)
                    
                # build feed_dict
                feed_dict = self.extra_params(
                        self.build_feed_dict(model.placeholders, bj), 
                        mode, lr)


                results = sess.run( fetch_data, 
                                    feed_dict = feed_dict)

                l = results[0]

                if visualizer:
                    if i % visualizer.interval == 0:
                        visualizer.train_log(results[-1], j)

                # accumulate loss
                avg_loss += l

            if verbose:
                log = '[{}] loss : {}'.format(i, 
                        avg_loss/(num_iterations))
                tqdm.write(log)

            # update lr
            #if i and i%25 == 0:
            #    lr = lr/2 # anneal

            
            # eval
            if eval_interval:
                if i and i%eval_interval == 0:
                    eloss, eacc = self.evaluate(visualizer=visualizer)
                    loss_trend.append(eloss)
                    accuracies.append(eacc)

                    # check if accuracy is better
                    if len(accuracies) > 2 and eacc > max(accuracies[:-1]):
                        # save model params
                        model_params = sess.run(tf.trainable_variables())

                    if early_stop_after and i > early_stop_after:
                        if early_stopping(loss_trend) or eacc > 0.95:
                            tqdm.write('stopping from early stopping')

                            # set best performing model params
                            if model_params:
                                print(':: Setting best model params')
                                sess.run(set_op(model_params))

                            # return max evaluation accuracy
                            return max(accuracies)

        # set best performing model params
        if model_params:
            sess.run(set_op(model_params))
        # end of epochs
        return max(accuracies)


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

    def extra_params(self, feed_dict, mode, lr=None):
        feed_dict[self.model.mode] = mode
        if lr:
            feed_dict[self.model.lr] = lr
        return feed_dict


def early_stopping(loss_trend):
    if len(loss_trend) < 4:
        return False
    
    for p, n in zip(loss_trend[-4:], loss_trend[-3:]):
        if p > n:
            return False
    return True
