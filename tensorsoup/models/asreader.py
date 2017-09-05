import sys
sys.path.append('../')

import tensorflow as tf

from sanity import sanity

cell = tf.nn.rnn_cell.LSTMCell
rnn = tf.nn.rnn_cell

clen = 20
qlen = 10
vocab_size = 100
batch_size = 2
max_candidates = 10
demb = 32
dhdim = 32
num_layers = 3


class ASReader(object):

    def __init__(self, clen, qlen, vocab_size, max_candidates,
            demb, dhdim, num_layers):

        # clear global graph
        tf.reset_default_graph()

        # set initializer
        self.init = tf.random_uniform_initializer(-0.1, 0.1)

        # num of copies of model (for multigpu training)
        self.n = 1

        # define placeholders
        self._context = tf.placeholder(tf.int32, [None, clen ], 
                                  name = 'context')
        self._query = tf.placeholder(tf.int32, [None, qlen],
                               name= 'query')
        self._answer = tf.placeholder(tf.int32, [None, ], 
                                name= 'answer')
        self._candidates = tf.placeholder(tf.int32, [None, max_candidates],
                                    name='candidates')
        self._cmask = tf.placeholder(tf.float32, [None, max_candidates, clen],
                               name='cmask')

        # default placeholders
        mode = tf.placeholder(tf.int32, shape=[], name='mode')
        lr = tf.placeholder(tf.float32, shape=[], name='lr')
 
        # get actual length of context and query
        batch_clen = tf.reduce_sum(tf.cast(self._context>0, tf.int32), axis=1)
        batch_qlen = tf.reduce_sum(tf.cast(self._query>0, tf.int32), axis=1)

        # setup embedding matrix
        emb = tf.get_variable('emb', [vocab_size, demb], tf.float32,
                                     initializer=self.init)

        # query encoder
        with tf.variable_scope('q_encoder', 
                               initializer=tf.orthogonal_initializer()):
            # lookup
            query_d = tf.nn.embedding_lookup(emb, self._query)
            # forward/backword cells
            fcell = rnn.MultiRNNCell([cell(dhdim) for _ in range(num_layers)])
            bcell = rnn.MultiRNNCell([cell(dhdim) for _ in range(num_layers)])
            outputs, q_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = fcell,
                                                              cell_bw = bcell,
                                                              inputs=query_d,
                                                              swap_memory=True,
                                                              sequence_length=batch_qlen,
                                                              dtype=tf.float32
                                                              )
            # combine layer_3 forward and backward states
            query_state = tf.concat([q_state[0][-1].c, q_state[1][-1].c], axis=-1)

        # context encoder
        with tf.variable_scope('c_encoder', 
                               initializer=tf.orthogonal_initializer()):
            # lookup
            context_d = tf.nn.embedding_lookup(emb, self._context)
            # forward/backword cells
            fcell = rnn.MultiRNNCell([cell(dhdim) for _ in range(num_layers)])
            bcell = rnn.MultiRNNCell([cell(dhdim) for _ in range(num_layers)])
            c_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fcell,
                                                              cell_bw = bcell,
                                                              inputs=context_d,
                                                              swap_memory=True,
                                                              sequence_length=batch_clen,
                                                              dtype=tf.float32
                                                                  )
            # combine forwared and backward states
            context = tf.concat(c_states, axis=-1)

        # calculate attention
        #  dot_product(context, query_state)
        attention = tf.matmul(tf.expand_dims(query_state, axis=-1),
                                     context, adjoint_a=True, adjoint_b=True)
        attention = tf.squeeze(attention)

        # sum up attention values of candidates
        attention_sum = tf.reduce_sum(
                tf.expand_dims(attention, axis=1)*self._cmask, axis=2)

        # normalize attention
        prob = tf.nn.softmax(attention_sum)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=attention_sum,
                labels=self._answer
                )
            # mean loss
            #loss = tf.reduce_mean(cross_entropy)
            loss = cross_entropy

        with tf.name_scope('evaluation'):
            correct_labels = tf.equal(tf.cast(self._answer, tf.int64), tf.argmax(prob, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))

        with tf.name_scope('optimization'):
            # gradient clipping
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            gvs = optimizer.compute_gradients(loss)
            clipped_gvs = [(tf.clip_by_norm(grad, 10.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(clipped_gvs)


        # attach to instance
        self.prob = prob
        self.attention = attention
        self.attention_sum = attention_sum
        self.loss = tf.reduce_mean(loss)
        self.accuracy = accuracy
        self.mode = mode
        self.lr = lr

        self.placeholders = [ self._context, self._query, 
                self._answer, self._candidates, self._cmask ]



if __name__ == '__main__':
    clen = 20
    qlen = 10
    vocab_size = 100
    max_candidates = 10
    demb = 32
    dhdim = 32
    num_layers = 3

    # instantiate model
    model = ASReader(clen, qlen, vocab_size, max_candidates, demb,
            dhdim, num_layers)

    results = sanity([model.attention, model.attention_sum, model.prob], 
            fetch_data=True)
