import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append('../../')


from recurrence import *
#from data.squad import SQuAD
from models.matchlstm.modules import match, answer_pointer
from sanity import *


class MatchLSTM():

    def __init__(self, emb_dim, hidden_dim, lr=0.0001):

        self.emb_dim = emb_dim
        self.d = hidden_dim
        self.lr = lr

        # clear graph
        tf.reset_default_graph()

        # initializer
        self.init = tf.random_normal_initializer(-0.08, 0.08)

        # placeholders
        self.passages = tf.placeholder(shape=[None, None, emb_dim], dtype=tf.float32, name='Passage')
        self.queries = tf.placeholder(shape=[None, None, emb_dim], dtype=tf.float32, name='Query')
        self.targets = tf.placeholder(shape=[2, None],
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
