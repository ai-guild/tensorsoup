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

        def tq(x):
            return tqdm(x) if verbose else x

        model = self.model
        datasrc = self.datasrc
        batch_size = self.datasrc.batch_size
        sess = self.sess

        # init sess
        sess.run(tf.global_variables_initializer())
        
        num_examples = datasrc.n['train']
        num_batches = int((num_examples/batch_size))
        
        for i in range(epochs):
            avg_loss, avg_acc = 0., 0.
            for j in tq(range(num_batches)):
                bj = datasrc.next_batch('train')
                l,acc, _ = sess.run( [model.loss, model.accuracy, model.train_op],
                                           feed_dict = self.build_feed_dict(model.placeholders, bj) )
                            
                # accumulate loss, accuracy
                avg_loss += l
                avg_acc += acc
            if verbose:
                log = '[{}] loss : {}; accuracy : {}'.format(i,
                        avg_loss/(num_batches), avg_acc/(num_batches))
                tqdm.write(log)
            
            # eval
            if i and i%eval_interval == 0:
                self.evaluate()


    def build_feed_dict(self, l1, l2):
        return { i:j for i,j in zip(l1,l2)}
