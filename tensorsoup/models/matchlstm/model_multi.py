import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append('../../')


from recurrence import *
from models.matchlstm.modules import match, answer_pointer
from train.utils import *


class MatchLSTM():

    def __init__(self, emb_dim, hidden_dim, lr=0.0001):

        self.emb_dim = emb_dim
        self.d = hidden_dim
        self.lr = lr

        # clear graph
        tf.reset_default_graph()

        # initializer
        self.init = tf.random_normal_initializer(-0.08, 0.08)

        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    '''
        Inference 

        (returns) - (logits, probability, argmax)
    '''
    def inference(self):

        # placeholders
        passages = tf.placeholder(shape=[None, None, self.emb_dim], 
                dtype=tf.float32, name='passages')
        queries = tf.placeholder(shape=[None, None, self.emb_dim], 
                dtype=tf.float32, name='queries')
        targets = tf.placeholder(shape=[2, None], 
                dtype=tf.int32, name='labels')
        masks = tf.placeholder(shape=[2, None, None] , 
                dtype=tf.float32, name='masks')

        # expose handle to placeholders
        placeholders = [ passages, queries, targets, masks ]

        # hidden dim
        d= self.d

        # LSTM Preprocessing Layer
        with tf.variable_scope('passage'):
            pcell = rcell('lstm', num_units=d)
            _, pstates = uni_net_dynamic(cell=pcell, inputs=passages, proj_dim=d)
        with tf.variable_scope('query'):
            qcell = rcell('lstm', d)
            _, qstates = uni_net_dynamic(cell=qcell, inputs=queries, proj_dim=d)


        # Match-LSTM Layer
        with tf.variable_scope('mlstm_forward'):
            _, states_f = match(qstates, pstates, d)

        with tf.variable_scope('mlstm_backward'):
            _, states_b = match(qstates, tf.reverse(pstates, axis=[0]), d)

        # concat forward and backward states
        mlstm_states = tf.concat([states_f, states_b], axis=-1, name='match_lstm_states')


        # Answer Pointer
        with tf.variable_scope('answer_pointer'):
            probs, logits = answer_pointer(mlstm_states, d, initializer=self.init)

        return logits, probs, placeholders


    def compute_loss(self, logits, targets, masks):

        # Loss/Cost
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=logits*masks, labels=targets))

        # L1 regularizer
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
        weights = tf.trainable_variables() # all vars of your graph
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        regularized_loss = ce + (1e-4*regularization_penalty)

        # Evaluation : accuracy
        probs = tf.nn.softmax(logits)
        correct_labels = tf.equal(tf.argmax(probs, axis=-1), tf.cast(targets, tf.int64))
        start_idx, end_idx = tf.unstack(tf.cast(correct_labels, dtype=tf.float32))
        accuracy = tf.reduce_mean(start_idx * end_idx)

        return regularized_loss, accuracy


    def make_parallel(self, num_copies, num_gpus):
        # num of copies of model
        self.n = num_copies

        # keep track of placholder and gradients
        tower_grads, ph, losses = [], [], []

        with tf.device('/cpu:0'):
            with tf.variable_scope(tf.get_variable_scope()):
                # number of copies
                for i in range(num_gpus):
                    with tf.device('/gpu:{}'.format(i)):
                        for j in range(num_copies//num_gpus):
                            with tf.name_scope('gpu_{}_{}'.format(i,j)) as scope:
                                # run inference
                                #  get handles to placeholders, logits and probs
                                logits, probs, placeholders = self.inference()
                                # pick targets, masks from placeholders, for loss/optimization
                                targets, masks = placeholders[2:]

                                # get loss, accuracy
                                loss, accuracy = self.compute_loss(logits, targets, masks)

                                # reuse trainable parameters
                                tf.get_variable_scope().reuse_variables()

                                # gather gradients
                                grads = self.opt.compute_gradients(loss)

                                # save grads for averaging later
                                tower_grads.append(grads)

                                # save the list of placholder handles
                                ph.append(placeholders)

                                # add loss
                                losses.append(loss)

            # average gradients
            grads = average_gradients(tower_grads)
            # apply averaged gradients
            apply_gradient_op = self.opt.apply_gradients(grads)

            # attach to instance
            self.train_op = apply_gradient_op
            # losses
            self.loss = losses

            # attach placeholders to instance
            self.placeholders = ph

        return 
