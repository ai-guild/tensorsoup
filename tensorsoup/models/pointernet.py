import tensorflow as tf


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
