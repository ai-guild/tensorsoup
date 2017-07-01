import tensorflow as tf
import sys

from init import *

'''
    Attention Mechanism

    based on "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://arxiv.org/abs/1409.0473

    [usage]
    ci = attention(enc_states, query, params= {
        'Wa' : Wa, # [d,d]
        'Ua' : Ua, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(enc_states) : [B, L, d] (batch_major)
    shape(query)  : [B, d]
    shape(ci)         : [B,d]

'''
@init
def attention(ref, query, d):

    # infer timesteps
    batch_size, timesteps, _ = tf.unstack(tf.shape(ref))

    Wa = tf.get_variable('Wa', shape=[d, d], dtype=tf.float32)#, initializer=def_init)
    Ua = tf.get_variable('Ua', shape=[d, d], dtype=tf.float32)#, initializer=def_init)
    Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)#, initializer=def_init)

    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(query, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(ref, [-1, d]), Ua), [-1, timesteps, d]))
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(ref*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    



'''
    Gated Attention Network

    based on "R-NET: Machine Reading Comprehension with Self-matching Networks"
        https://www.microsoft.com/en-us/research/publication/mrc/

    [usage]
    dec_outputs, dec_states = gated_attention_net(enc_states, # encoded representation of text
                                    tf.zeros(dtype=tf.float32, shape=[B,d*2]), # notice d*2
                                    batch_size=B,timesteps=L,feed_previous=False,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]

    For reading comprehension, inputs is same as enc_states; feed_previous doesn't apply


'''
def gated_attention_net(enc_states, init_state, batch_size, 
                      d, timesteps,
                      inputs = [],
                      scope='gated_attention_net_0',
                      feed_previous=False):
    
    # define attention parameters
    Wa = tf.get_variable('Wa', shape=[d*2, d], dtype=tf.float32)
    Ua = tf.get_variable('Ua', shape=[d, d], dtype=tf.float32)
    Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)
    att_params = {
        'Wa' : Wa, 'Ua' : Ua, 'Va' : Va
    }
    
    # define rnn cell
    cell = gru(num_units=d*2)
    
    # gate params
    Wg = tf.get_variable('Wg', shape=[d*2, d*2], dtype=tf.float32)
        
    def step(input_, state):
        # define input gate
        gi = tf.nn.sigmoid(tf.matmul(input_, Wg))
        # apply gate to input
        input_ = gi * input_
        # recurrent step
        output, state = cell(input_, state)
        return output, state
    
    outputs = [inputs[0]] # include GO token as init input
    states = [init_state]
    for i in range(timesteps):
        if i>0:
            tf.get_variable_scope().reuse_variables()

        input_ = outputs[-1] if feed_previous else inputs[i]

        # get match for current word
        ci = attention(enc_states, states[-1], att_params, d, timesteps)
        # combine ci and input(i) 
        input_ = tf.concat([input_, ci], axis=-1)
        output, state = step(input_, states[-1])
    
        outputs.append(output)
        states.append(state)

    # time major -> batch major
    states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2])
    outputs_bm = tf.transpose(tf.stack(outputs[1:]), [1, 0, 2])

    return outputs_bm, states_bm


'''
    Attention Mechanism for Pointer Network

    based on "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://arxiv.org/abs/1409.0473

    [usage]
    ci = attention(enc_states, dec_state, params= {
        'Wa' : Wa, # [d,d]
        'Ua' : Ua, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(enc_states) : [B, L, d]
    shape(dec_state)  : [B, d]
    shape(ci)         : [B,d]

'''
def ptr_attention(enc_states, dec_state, params, normalize=False):

    # infer shapes from tensors
    de = tf.shape(enc_states)[-1]
    timesteps = tf.shape(enc_states)[1]
    d = tf.shape(dec_state)[-1]

    Wa, Ua = params['Wa'], params['Ua']
    # s_j -> [B,L,d]
    a = tf.nn.elu(tf.expand_dims(tf.matmul(dec_state, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(enc_states,[-1, de]), Ua), [-1, timesteps, d]))
    Va = params['Va'] # [d, 1]
    # e_j -> [B, L]
    scores = tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps])

    if normalize:
        return tf.nn.softmax(scores)

    return scores


'''
    Pointer decoder

    based on "Pointer Networks"
        https://arxiv.org/abs/1506.03134


    [usage]
'''
def pointer_decoder(estates, as_prob=False, feed_previous=False):
    # special generation symbol
    special_sym_value = 20.
    special_sym = tf.constant(special_sym_value, shape=[batch_size, 1], dtype=tf.float32)
    # decoder states
    dec_init_state = estates[-1]
    dstates = [dec_init_state]
    # decoder input
    d_input_ = special_sym

    # create cell
    dcell = gru(d)

    logits = []
    probs = []
    dec_outputs = []
    for i in range(num_indices):
        if i>0:
            tf.get_variable_scope().reuse_variables()
            
        # project input
        Wp = tf.get_variable('W_p', [1, d], initializer=init)
        bp = tf.get_variable('b_p', [d], initializer=init)
        
        d_input_ = tf.nn.elu(tf.matmul(d_input_, Wp) + bp, name='decoder_cell_input')
        
        # step
        output, dec_state = dcell(d_input_, dstates[-1])
        
        # project enc/dec states
        W1 = tf.get_variable('W_1', [d, d], initializer=init)
        W2 = tf.get_variable('W_2', [d, d], initializer=init)
        v = tf.get_variable('v', [d, 1], initializer=init)
        
        # pass encoder states as batch_major
        scores = ptr_attention(tf.transpose(estates, [1,0,2]), dec_state,
                      params = {'Wa' : W1, 'Ua' : W2, 'Va' : v}, d = d, timesteps=L)
        
        prob_dist = tf.nn.softmax(scores)
        idx = tf.argmax(prob_dist, axis=1)
        
        # get input at index "idx"
        dec_output_i = batch_gather_nd(inputs, idx)
        
        if feed_previous:
            # output at i is input to i+1
            d_input_ = tf.expand_dims(dec_output_i, axis=-1)
        else:
            idx = tf.argmax(targets[i], axis=1)
            d_input_ = tf.expand_dims(batch_gather_nd(inputs, idx), axis=-1)
        
        logits.append(scores)
        probs.append(prob_dist)

        dec_outputs.append(dec_output_i)
    
    if as_prob:
        return dec_outputs, tf.stack(probs)

    return dec_outputs, tf.stack(logits)
