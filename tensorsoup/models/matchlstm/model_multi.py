import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.append('../../')


from recurrence import *
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


    def loss(self, logits, targets, masks):

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



if __name__ == '__main__':

    
    d = 150
    model = MatchLSTM(emb_dim=100, hidden_dim=d, lr=0.0001)
    if sanity([model.loss, model.accuracy]):
        print('Sanity check successful!')

    '''
    print('Preparing data ...')
    squad_ = SQuAD(datadir='../../../datasets/SQuAD/', 
                    glove_file='../../../datasets/glove/glove.6B.100d.txt')

    print('Initializing Model ...')
    model = MatchLSTM(emb_dim=100, hidden_dim=d, lr=0.0001)

    try:
        print('Starting Training ...')
        model.train(squad_, batch_size=128, epochs=2000)
    except KeyboardInterrupt:
        model.f.close()
        model.summarize()
    '''
