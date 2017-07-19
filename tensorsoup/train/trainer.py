from tqdm import tqdm
import tensorflow as tf


class Trainer(object):

    def __init__(self, sess, model, datasrc, batch_size):
        self.model = model
        self.datasrc = datasrc
        self.sess = sess


    def evaluate(self):
        datasrc = self.datasrc
        batch_size = datasrc.batch_size
        num_examples = datasrc.n['test']
        model = self.model

        # num copies
        n = model.n

        num_iterations = int(num_examples / (batch_size*n))

        build_feed = self.build_feed_dict if n == 1 else self.build_feed_dict_multi

        avg_loss, avg_acc = 0., 0.
        datasrc.i = 0 # point to zero index for test
        for i in tqdm(range(num_iterations)):
            bi = datasrc.next_batch(n, 'test')
            l, acc = self.sess.run( [model.loss, model.accuracy],
                            feed_dict = build_feed(model.placeholders, bi))
            avg_loss += l
            avg_acc += acc

        log = 'Evaluation - loss : {}; accuracy : {}'.format(avg_loss/(num_iterations),
                            avg_acc/(num_iterations))
        tqdm.write(log)


    def fit(self, epochs, eval_interval=10, verbose=True):

        def tq(x):
            return tqdm(x) if verbose else x

        model = self.model
        datasrc = self.datasrc
        batch_size = self.datasrc.batch_size
        sess = self.sess

        # num copies of model
        n = model.n

        # init sess
        sess.run(tf.global_variables_initializer())
        
        num_examples = datasrc.n['train']
        num_iterations = int(num_examples/(batch_size*n))

        build_feed = self.build_feed_dict if n == 1 else self.build_feed_dict_multi
        
        for i in range(epochs):
            avg_loss, avg_acc = 0., 0.
            datasrc.i = 0 # point to start index
            for j in tq(range(num_iterations)):
                bj = datasrc.next_batch(n, 'train')
                l, acc, _ = sess.run( [model.loss, model.accuracy, model.train_op],
                                           feed_dict = build_feed(model.placeholders, bj) )

                # accumulate loss, accuracy
                avg_loss += l
                avg_acc += acc

            if verbose:
                log = '[{}] loss : {}; accuracy : {}'.format(i,
                        avg_loss/(num_iterations), avg_acc/(num_iterations))
                tqdm.write(log)
            
            # eval
            if i and i%eval_interval == 0:
                self.evaluate()


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
