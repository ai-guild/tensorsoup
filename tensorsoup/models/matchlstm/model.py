import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append('../../')


from recurrence import *
#from data.squad import SQuAD
from models.matchlstm.modules import match, answer_pointer
from sanity import *


class MatchLSTM():

    def __init__(self, emb_dim, hidden_dim, num_indices=2, lr=0.0001):

        self.emb_dim = emb_dim
        self.d = hidden_dim
        self.num_indices = num_indices
        self.lr = lr

        # clear graph
        tf.reset_default_graph()

        # initializer
        self.init = tf.random_normal_initializer(-0.08, 0.08)

        # placeholders
        self.passages = tf.placeholder(shape=[None, None, emb_dim], dtype=tf.float32, name='Passage')
        self.queries = tf.placeholder(shape=[None, None, emb_dim], dtype=tf.float32, name='Query')
        self.targets = tf.placeholder(shape=[num_indices, None],
                        dtype=tf.int32, name='labels')
        self.masks = tf.placeholder(shape=[2, None, None] , dtype=tf.float32, name='masks')


        # infer batch_size
        self.batch_size, _, _ = tf.unstack(tf.shape(self.passages))


        # LSTM Preprocessing Layer
        with tf.variable_scope('passage'):
            pcell = rcell('lstm', num_units=self.d)
            _, pstates = uni_net_dynamic(cell=pcell, inputs=self.passages, proj_dim=self.d)
        with tf.variable_scope('query'):
            qcell = rcell('lstm', self.d)
            _, qstates = uni_net_dynamic(cell=qcell, inputs=self.queries, proj_dim=self.d)


        # Match-LSTM Layer
        with tf.variable_scope('mlstm_forward'):
            _, states_f = match(qstates, pstates, self.d)

        with tf.variable_scope('mlstm_backward'):
            _, states_b = match(qstates, tf.reverse(pstates, axis=[0]), self.d)

        # concat forward and backward states
        mlstm_states = tf.concat([states_f, states_b], axis=-1, name='match_lstm_states')


        # Answer Pointer
        with tf.variable_scope('answer_pointer'):
            probs, logits = answer_pointer(mlstm_states, self.d, initializer=self.init)

        # Loss/Cost
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=logits*self.masks, labels=self.targets))

        # L1 regularizer
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        regularized_loss = ce + (1e-4*regularization_penalty)

        self.loss = regularized_loss


        # Evaluation : accuracy
        correct_labels = tf.equal(tf.argmax(probs, axis=-1), tf.cast(self.targets, tf.int64))
        start_idx, end_idx = tf.unstack(tf.cast(correct_labels, dtype=tf.float32))
        self.accuracy = tf.reduce_mean(start_idx * end_idx)

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # gradient clipping
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_norm(grad, 2.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(clipped_gvs)

        # create session
        # self.sess = tf.Session()

        # placholders as ordered list
        self._placeholders()

    
    def _placeholders(self):
        self.placeholders = [ self.passages, self.queries, self.targets, self.masks ]


    def train(self, squad, batch_size=128, epochs=1000, eval_interval=10):

        # init sess
        self.sess.run(tf.global_variables_initializer())
        
        num_examples = squad.n['train']
        num_batches = int((num_examples/batch_size)/8)
        for i in range(epochs):
            avg_loss, avg_acc = 0., 0.
            nan_count = 0
            for j in tqdm(range(num_batches)):
                pj, qj, tj, mj = squad.next_batch(batch_size, 'train')
                l, acc, _ = self.sess.run( [self.loss, self.accuracy, self.train_op],
                                feed_dict = {
                                    self.passages : pj,
                                    self.queries : qj,
                                    self.targets : tj,
                                    self.masks : mj
                                    })
                # accumulate loss, accuracy
                avg_loss += l
                avg_acc += acc

            # print info
            log = '[{}] loss : {}; accuracy : {}'.format(i,
                    avg_loss/(num_batches), avg_acc/(num_batches))
            # log message
            tqdm.write(log)

            # eval
            if i and i%eval_interval == 0:
                self.evaluate(squad)


    def evaluate(self, squad):

        batch_size = 128 
        num_examples = squad.n['dev']
        num_batches = num_examples // batch_size

        avg_loss, avg_acc = 0., 0.
        nan_count = 0
        for i in tqdm(range(num_batches)):
            pi, qi, ti, mi = squad.batch('dev', i,  batch_size)
            l, acc = self.sess.run( [self.loss, self.accuracy],
                            feed_dict = {
                                self.passages : pi,
                                self.queries : qi,
                                self.targets : ti,
                                self.masks : mi
                                })
            # accumulate loss, accuracy
            if l < 10 and l > 0:
                avg_loss += l
                avg_acc += acc
            else:
                nan_count += 1
                

        # print results
        try:
            log = 'Evaluation - loss : {}; accuracy : {}'.format(avg_loss/(num_batches-nan_count),
                            avg_acc/(num_batches-nan_count))
        except:
            pass


if __name__ == '__main__':

    
    d = 150
    num_indices = 2
    model = MatchLSTM(emb_dim=100, hidden_dim=d, num_indices=2, lr=0.0001)
    if sanity([model.loss, model.accuracy]):
        print('Sanity check successful!')

    '''
    print('Preparing data ...')
    squad_ = SQuAD(datadir='../../../datasets/SQuAD/', 
                    glove_file='../../../datasets/glove/glove.6B.100d.txt')

    print('Initializing Model ...')
    model = MatchLSTM(emb_dim=100, hidden_dim=d, num_indices=2, lr=0.0001)

    try:
        print('Starting Training ...')
        model.train(squad_, batch_size=128, epochs=2000)
    except KeyboardInterrupt:
        model.f.close()
        model.summarize()
    '''
