import tensorflow as tf


'''
    Attention Pooling Mechanism

    based on "R-NET: Machine Reading Comprehension with Self-matching Networks"
        https://www.microsoft.com/en-us/research/publication/mrc/

    [usage]
    ci = attention(enc_states, dec_state, params= {
        'Wa' : Wa, # [d,d]
        'Wb' : Wb, # [d,d]
        'Wc' : Wc, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(states_a) : [B, L, d]
    shape(states_b_i) : [B, d]
    shape(state_c)    : [B, d]
    shape(ci)         : [B, d]

'''
def attention_pooling(states_a, states_b_i, state_c, params, d, timesteps):
    Wa, Wb, Wc = params['Wa'], params['Wb'], params['Wc']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(states_b_i, Wb), axis=1) +
            tf.reshape(tf.matmul(tf.reshape(states_a,[-1, d]), Wa), [-1, timesteps, d]) +
            tf.expand_dims(tf.matmul(state_c, Wc)))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(enc_states*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    


