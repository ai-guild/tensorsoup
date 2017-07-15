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
        num_batches = num_examples // batch_size
        model = self.model

        avg_loss, avg_acc = 0., 0.
        for i in tqdm(range(num_batches)):
            bi = datasrc.batch(i, 'test')
            l, acc = self.sess.run( [model.loss, model.accuracy],
                            feed_dict = self.build_feed_dict(model.placeholders, bi))
            avg_loss += l
            avg_acc += acc

        log = 'Evaluation - loss : {}; accuracy : {}'.format(avg_loss/(num_batches),
                            avg_acc/(num_batches))
        tqdm.write(log)


    def fit(self, epochs, eval_interval=10, verbose=True):

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
        
        for i in range(epochs):
            avg_loss, avg_acc = 0., 0.
            for j in tqdm(range(num_iterations)):
                bj = datasrc.next_n_batches(n, 'train')
                l, _ = sess.run( [model.loss, model.train_op],
                                           feed_dict = self.build_feed_dict(model.placeholders, bj) )
                tqdm.write(str(sum(l)/len(l)))
                            
            '''
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
            '''

    def build_feed_dict(self, ll1, ll2):
        feed_dict = {}
        # we have a tuple
        #  [1] list of list of placeholders
        #  [2] list of list of data in batches
        for l1, l2 in zip(ll1, ll2):
            for i,j in zip(l1,l2):
                feed_dict[i] = j

        return feed_dict
