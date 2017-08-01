import tensorflow as tf
import numpy as np

import sys
sys.path.append('../../')

from attention import attention
from sanity import *

from collections import OrderedDict


class MemoryNet():

    def __init__(self, hdim, num_hops, memsize, 
            window_size, sentence_size, vocab_size, 
            lr1=0.025, lr2=0.025):

        # reset graph
        tf.reset_default_graph()

        # initializer
        self.init = tf.random_normal_initializer(0, 0.1)

        # num of copies of model (for multigpu training)
        self.n = 1

        #optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=lr1)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=lr2)


        def inference():

            with tf.name_scope('input'):

                # placeholders
                queries = tf.placeholder(tf.int32, shape=[None, sentence_size], 
                        name='queries')
                windows = tf.placeholder(tf.int32, shape=[None, memsize, window_size], 
                        name='windows')
                answers = tf.placeholder(tf.int32, shape=[None, ], 
                        name='answers')
                candidates = tf.placeholder(tf.int32, shape=[None, 10], 
                        name='candidates')
                window_targets = tf.placeholder(tf.int32, shape=[None, memsize],
                        name='window_targets')
                mode = tf.placeholder(tf.int32, shape=(), name='mode')

            # expose handle to placeholders
            placeholders = OrderedDict()
            placeholders['queries'] = queries
            placeholders['windows'] = windows
            placeholders['answers'] = answers
            placeholders['candidates'] = candidates
            placeholders['window_targets'] = window_targets

            # embedding
            with tf.name_scope('embeddings'):
                A = tf.get_variable('A', shape=[vocab_size, hdim], dtype=tf.float32, 
                                   initializer=self.init)
                #B = tf.get_variable('B', shape=[vocab_size, hdim], dtype=tf.float32, 
                #                   initializer=self.init)
                C = tf.get_variable('C', shape=[vocab_size, hdim], dtype=tf.float32, 
                                   initializer=self.init)

            # NOTE : remove position encoding
            # Position encoding
            # encoding = tf.constant(self.position_encoding(sentence_size, hdim))

            with tf.name_scope('question'):
                # Embed Questions (A)
                u0 = tf.nn.embedding_lookup(A, queries)
                # NOTE : remove position encoding
                # u0 = tf.reduce_sum(u0 * encoding, axis=1)
                u0 = tf.reduce_sum(u0, axis=1)
                u = [u0] # accumulate question emb

            with tf.name_scope('temporal'):
                # more variables
                H = tf.get_variable('H', dtype=tf.float32, shape=[hdim, hdim],
                        initializer=self.init)
                TA = tf.get_variable('TA', dtype=tf.float32, shape=[num_hops, memsize, hdim],
                        initializer=self.init)
                TC = tf.get_variable('TC', dtype=tf.float32, shape=[num_hops, memsize, hdim],
                        initializer=self.init)

            with tf.name_scope('memloop'):
                # memory loop
                for i in range(num_hops):
                    # embed windows
                    m = tf.nn.embedding_lookup(A, windows)
                    # NOTE : remove position encoding
                    # m = tf.reduce_sum(m*encoding, axis=2) + TA[i]
                    m = tf.reduce_sum(m, axis=2) + TA[i]
                    c = tf.nn.embedding_lookup(C, windows)
                    c = tf.reduce_sum(c, axis=2) + TC[i]

                    score = tf.reduce_sum(m*tf.expand_dims(u[-1], axis=1), axis=-1)
                    p = tf.cond(mode>0,
                                lambda : tf.nn.softmax(score),
                                lambda : score)
                    o = tf.reduce_sum(tf.expand_dims(p, axis=-1)*c, axis=1)
                    u_k = tf.matmul(u[-1], H) + o
                    u.append(u_k)


            with tf.name_scope('answer'):
                cand_emb = tf.nn.embedding_lookup(A, candidates)
                logits = attention(cand_emb, u[-1], d=hdim, score=True,
                        initializer=self.init)
                probs = tf.nn.softmax(logits)
                # prediction
                self.prediction = tf.argmax(probs)

            with tf.name_scope('loss'):
                # optimization
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=answers
                    )
                # attach loss to instance
                loss = tf.reduce_mean(cross_entropy)

            # memory access supervision
            with tf.name_scope('self_sup'):
                # we have 'scores' of memories
                ma_loss = tf.losses.mean_squared_error(window_targets, 
                        tf.nn.softmax(score))

            with tf.name_scope('evaluation'):
                # evaluation
                correct_labels = tf.equal(tf.cast(answers, tf.int64), tf.argmax(probs, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))

            # optimization
            with tf.name_scope('optimization'):
                # gradient clipping
                gvs = optimizer1.compute_gradients(loss)
                clipped_gvs = [(tf.clip_by_norm(grad, 40.), var) for grad, var in gvs]
                train_op_primary = optimizer1.apply_gradients(clipped_gvs)

                # optimize memory access
                train_op_ma = optimizer2.minimize(ma_loss)

                self.train_op = [ train_op_primary, train_op_ma ]


            self.loss = loss + ma_loss
            self.accuracy = accuracy
            self.windows = windows
            self.queries = queries
            self.answers = answers
            self.candidates = candidates
            self.window_targets = window_targets
            self.mode = mode
            
            self.placeholders = [queries, windows, answers, candidates, window_targets]

        # execute and build graph
        inference()


    def position_encoding(self, sentence_size, embedding_size):

        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size+1
        le = embedding_size+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        
        return np.transpose(encoding)


    '''
        clean up and organize the methods below

    def hard_mem_access(self, probs, c):
        m_max_id = tf.cast(tf.argmax(probs, axis=1), dtype=tf.int32)
        range_ = tf.range(start=0, limit=tf.shape(m_max_id)[0])
        m_idx = tf.stack([range_, m_max_id], axis=1)
        return tf.gather_nd(c, m_idx)

    def soft_mem_access(self, probs, c):
        #c_o = tf.transpose(c, [0, 2, 1])*probs
        c_o = tf.transpose(c, [0, 2, 1])*tf.expand_dims(probs, axis=1)
        return tf.reduce_sum(c_o, axis=-1)

    def self_supervision(self, match_scores):
        # Self-Supervision
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=match_scores, labels=self.window_targets)
        loss = tf.reduce_mean(loss)
        mem_train_op = self.mem_optimizer.minimize(loss)
        # attach to class
        self.mem_train_op = mem_train_op
        self.self_sup_loss = loss

    def answer_select(self, C, W, u):
        cand_emb = tf.nn.embedding_lookup(C, self.candidates)
        u_proj = tf.matmul(u, W)
        bilinear = cand_emb * tf.expand_dims(u_proj, axis=1)
        logits = tf.reduce_sum(bilinear, axis=2)
        a = tf.nn.softmax(logits)
        return logits, a

    def optimize(self, logits, lr):
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=self.answers)
        self.loss = tf.reduce_mean(ce_loss)
        self.train_op = self.optimizer.minimize(self.loss)
    '''


if __name__ == '__main__':

    # create model
    print('> Create model')
    model = MemoryNet(hdim= 20, # embedding dimension
                             num_hops=3,
                             memsize=5,
                             sentence_size= 10, # max sentence len
                             vocab_size= 100,
                             lr1=0.025,
                             lr2=0.025
                             )
    print(sanity([model.train_op], fetch_data=True))
    # sanity check : success
