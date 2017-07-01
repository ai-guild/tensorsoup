import tensorflow as tf


'''
    Attention Mechanism for Match-LSTM

    based on "Learning Natural Language Inference with LSTM"
        https://arxiv.org/abs/1512.08849


    [usage]

'''
def mlstm_attention(states_a, states_b_i, state_c, params):
    # convert states_a to batch_major
    states_a_bm = tf.transpose(states_a, [1,0,2], name='a_batch_major')

    # infer hidden dim and timesteps (question length)
    _, qlen, d = tf.unstack(tf.shape(states_a_bm))
     
    Wa, Wb, Wc = params['Wa'], params['Wb'], params['Wc']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(states_b_i, Wb), axis=1) +
            tf.reshape(tf.matmul(tf.reshape(states_a_bm,[-1, d]), Wa), [-1, qlen, d]) +
            tf.expand_dims(tf.matmul(state_c, Wc), axis=1))

    # calculate match/score
    Va = params['Va'] # [d, 1]
    b = params['ba']
    scores = tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, qlen]) + b
    probs = tf.nn.softmax(scores)
    
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(states_a_bm*tf.expand_dims(probs, axis=-1), axis=1) # [B, d]    


'''
    Match LSTM

    based on "Learning Natural Language Inference with LSTM"
        https://arxiv.org/abs/1512.08849


    [usage]

'''
#with tf.variable_scope('match_lstm'):
def mlstm(qstates, pstates, d):

    # infer batch_size, passage length and question length
    qlen, batch_size, _ = tf.unstack(tf.shape(qstates))
    plen = tf.shape(pstates)[0]
    
    # define attention parameters
    Wa = tf.get_variable('Wa', shape=[d, d], dtype=tf.float32)
    Wb = tf.get_variable('Wb', shape=[d, d], dtype=tf.float32)
    Wc = tf.get_variable('Wc', shape=[2*d, d], dtype=tf.float32)
    Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)
    Ba = tf.get_variable('ba', shape=(), dtype=tf.float32)
    att_params = {
        'Wa' : Wa, 'Wb' : Wb, 'Wc' : Wc, 'Va' : Va, 'ba' : Ba
    }
    
    # ouput projection params
    # Wo = tf.get_variable('Wo', shape=[2*d, d], dtype=tf.float32)
    
    # define rnn cell
    #  TODO : replace with LSTM
    cell = lstm(num_units=2*d)

    states = tf.TensorArray(dtype=tf.float32, size=plen+1, name='states',
                clear_after_read=False)
    outputs = tf.TensorArray(dtype=tf.float32, size=plen, name='outputs',
                clear_after_read=False)

    # set init state
    #init_state = tf.zeros(dtype=tf.float32, shape=[batch_size, 2*d])
    init_state = cell.zero_state(batch_size, tf.float32)
    states = states.write(0, init_state)

    def mlstm_step(i, states, outputs):
        # get previous state
        prev_state = states.read(i)

        prev_state = tf.unstack(prev_state)
        prev_state_tuple = rnn.LSTMStateTuple(prev_state[0], prev_state[1])
        prev_state_c = prev_state_tuple.c

        # get attention weighted representation
        ci = mlstm_attention(qstates, pstates[i], 
                prev_state_c, att_params)

        # combine ci and input(i) 
        input_ = tf.concat([pstates[i], ci], axis=-1)
        output, state = cell(input_, prev_state_tuple)

        # save output, state 
        states = states.write(i+1, state)
        outputs = outputs.write(i, output)

        return (i+1, states, outputs)

    # execute loop
    #i = tf.constant(0)
    c = lambda x, y, z : tf.less(x, plen)
    b = lambda x, y, z : mlstm_step(x, y, z)
    _, fstates, foutputs = tf.while_loop(c,b, [0, states, outputs])

    return foutputs.stack(), project_lstm_states(fstates.stack()[1:], 4*d, d)
