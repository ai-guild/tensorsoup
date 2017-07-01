import tensorflow as tf

import sys

sys.path.append('../')
from recurrence import *
from misc import *
from attention import *

def naive_seq2seq(inputs, targets, emb, ecell, dcell, training=True, feed_previous=True):

    batch_size, timesteps, _ = tf.unstack(tf.shape(inputs))
    d = ecell.state_size[0] if is_lstm(ecell) else ecell.state_size

    # encoder
    #  cell : ecell
    #  inputs : shape-> [B, L, enc_d]
    #  returns : encoder outputs, states
    with tf.variable_scope('encoder'):
        # uni-directional RNN
        enc_outputs, enc_states = uni_net_dynamic(ecell, inputs)

    # decoder
    #  cell : dcell
    #  targets : [
    with tf.variable_scope('decoder'):
        # GO token
        start_token = tf.zeros(dtype=tf.int64, shape=[batch_size, ]) + 1
        start_token_emb = tf.nn.embedding_lookup(emb, start_token)
        end_token = tf.zeros(dtype=tf.int64, shape=[batch_size,]) + 2
        # decoder
        targets_proj = tf.nn.embedding_lookup(emb, targets)
        dec_outputs = naive_decoder(dcell, enc_states, targets_proj, start_token_emb, 
                end_token, training=training, feed_previous=feed_previous)


def naive_decoder(cell, enc_states, targets, start_token, end_token, 
        feed_previous=True, training=True, scope='naive_decoder.0'):

    init_state = enc_states[-1]
    timesteps = tf.shape(enc_states)[0]

    # targets time major
    targets_tm = tf.transpose(targets, [1,0,2])

    states = tf.TensorArray(dtype=tf.float32, size=timesteps+1, name='states',
                    clear_after_read=False)
    outputs = tf.TensorArray(dtype=tf.float32, size=timesteps+1, name='outputs',
                    clear_after_read=False)

    def step(i, states, outputs):
        # run one step
        #  read from TensorArray (states)
        state_prev = states.read(i)

        if is_lstm(cell):
            # previous state <tensor> -> <LSTMStateTuple>
            c, h = tf.unstack(state_prev)
            state_prev = rnn.LSTMStateTuple(c,h)

        if feed_previous:
            input_ = outputs.read(i)
        else:
            input_ = targets_tm[i]

        output, state = cell(input_, state_prev)
        # add state, output to list
        states = states.write(i+1, state)
        outputs = outputs.write(i+1, output)
        i = tf.add(i,1)
        return i, states, outputs


    with tf.variable_scope(scope):
        # initial state
        states = states.write(0, init_state)
        # initial input
        outputs = outputs.write(0, start_token)

        i = tf.constant(0)

        # Stop loop condition
        if training:
            c = lambda x, y, z : tf.less(x, timesteps)
        else:
            c = lambda x, y, z : tf.reduce_all(tf.not_equal(tf.argmax(z.read(x), axis=-1), 
                    end_token))
        # body
        b = lambda x, y, z : step(x, y, z)
        # execution 
        _, fstates, foutputs = tf.while_loop(c,b, [i, states, outputs])

    return foutputs.stack()[1:] # add states; but why?


def attentive_seq2seq(inputs, targets, emb, ecell, training=True, feed_previous=True):

    batch_size, timesteps, _ = tf.unstack(tf.shape(inputs))
    d = ecell.state_size[0] if is_lstm(ecell) else ecell.state_size

    # encoder
    #  cell : ecell
    #  inputs : shape-> [B, L, enc_d]
    #  returns : encoder outputs, states
    with tf.variable_scope('encoder'):
        # uni-directional RNN
        enc_outputs, enc_states = uni_net_dynamic(ecell, inputs)

    # decoder
    #  cell : dcell
    #  targets : [
    with tf.variable_scope('decoder'):
        # GO token
        start_token = tf.zeros(dtype=tf.int64, shape=[batch_size, ]) + 1
        start_token_emb = tf.nn.embedding_lookup(emb, start_token)
        end_token = tf.zeros(dtype=tf.int64, shape=[batch_size,]) + 2
        # decoder
        targets_proj = tf.nn.embedding_lookup(emb, targets)
        dec_outputs = attentive_decoder(enc_states, targets_proj, d, start_token_emb, 
                end_token, training=training, feed_previous=feed_previous)


def attentive_decoder(enc_states, targets, hdim, start_token, end_token, 
        feed_previous=True, training=True, scope='attentive_decoder.0'):

    # hidden dimension
    d = hdim

    init_state = enc_states[-1]
    timesteps = tf.shape(enc_states)[0]

    # targets time major
    targets_tm = tf.transpose(targets, [1,0,2])

    states = tf.TensorArray(dtype=tf.float32, size=timesteps+1, name='states',
                    clear_after_read=False)
    outputs = tf.TensorArray(dtype=tf.float32, size=timesteps+1, name='outputs',
                    clear_after_read=False)

    def attentive_cell(input_, state, ci):
        # init a shit ton of variables
        U = get_variables(4, [d,d], name='U')
        W = get_variables(4, [d,d], name='W')
        C = get_variables(4, [d,d], name='C')

        z = tf.nn.sigmoid(tf.matmul(input_, W[1])+tf.matmul(state, U[1])+tf.matmul(ci, C[1]))
        r = tf.nn.sigmoid(tf.matmul(input_, W[2])+tf.matmul(state, U[2])+tf.matmul(ci, C[2]))
        si = tf.nn.tanh(tf.matmul(input_, W[0])+tf.matmul(ci, C[0])+tf.matmul(r*state, U[0]))
        
        state = (1-z)*state + z*si
        output = tf.matmul(state, U[3]) + tf.matmul(input_, W[3]) + tf.matmul(ci, C[3])
        
        return output, state
 
    def attentive_step(i, states, outputs):
        # run one step
        #  read from TensorArray (states)
        state_prev = states.read(i)

        if feed_previous:
            input_ = outputs.read(i)
        else:
            input_ = targets_tm[i]

        output, state = attentive_cell(input_, state_prev, 
                    attention(enc_states, state_prev, d))

        # add state, output to list
        states = states.write(i+1, state)
        outputs = outputs.write(i+1, output)
        i = tf.add(i,1)

        return i, states, outputs


    with tf.variable_scope(scope):
        # initial state
        states = states.write(0, init_state)
        # initial input
        outputs = outputs.write(0, start_token)

        i = tf.constant(0)

        # Stop loop condition
        if training:
            c = lambda x, y, z : tf.less(x, timesteps)
        else:
            c = lambda x, y, z : tf.reduce_all(tf.not_equal(tf.argmax(z.read(x), axis=-1), 
                    end_token))
        # body
        b = lambda x, y, z : attentive_step(x, y, z)
        # execution 
        _, fstates, foutputs = tf.while_loop(c,b, [i, states, outputs])

    return foutputs.stack()[1:] # add states; but why?
