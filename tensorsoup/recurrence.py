import tensorflow as tf

# alias
rnn = tf.nn.rnn_cell


def rcell(cell_type, num_units, num_layers=None, dropout=None):

    if cell_type == 'lstm':
        cell_ = rnn.LSTMCell(num_units)
    elif cell_type == 'gru':
        cell_ = rnn.GRUCell(num_units)

    if dropout is not None:
        _cell = lambda : rnn.DropoutWrapper(cell_, output_keep_prob=1. - dropout)

    if num_layers:
        cell_ = rnn.MultiRNNCell([_cell() for _ in range(num_layers)])

    return cell_



'''
    Uni-directional RNN

    [usage]
    cell_ = gru_n(hdim, 3)
    outputs, states = uni_net_static(cell = cell_,
                             inputs= inputs_emb,
                             init_state= cell_.zero_state(batch_size, tf.float32),
                             timesteps = L)
'''
def uni_net_static(cell, inputs, init_state, timesteps, time_major=False, scope='uni_net_0'):
    # convert to time major format
    if not time_major:
        inputs_tm = tf.transpose(inputs, [1, 0, 2])
    # collection of states and outputs
    states, outputs = [init_state], []

    with tf.variable_scope(scope):

        for i in range(timesteps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs_tm[i], states[-1])
            outputs.append(output)
            states.append(state)

    return tf.stack(outputs), tf.stack(states[1:])



def uni_net_dynamic(cell, inputs, proj_dim=None, init_state=None, scope='uni_net_d0'):
    # transpose to time major
    inputs_tm = tf.transpose(inputs, [1,0,2], name='inputs_tm')

    # infer timesteps and batch_size
    timesteps, batch_size, _ = tf.unstack(tf.shape(inputs_tm))

    # check if init_state is provided
    #  TODO : fix and add this
    # init_state = init_state if init_state else cell.zero_state(batch_size,tf.float32)
    if init_state is None:
        init_state = cell.zero_state(batch_size, tf.float32)

    states = tf.TensorArray(dtype=tf.float32, size=timesteps+1, name='states',
                    clear_after_read=False)
    outputs = tf.TensorArray(dtype=tf.float32, size=timesteps, name='outputs',
                    clear_after_read=False)

    def step(i, states, outputs):
        # run one step
        #  read from TensorArray (states)
        state_prev = states.read(i)

        if is_lstm(cell):
            # previous state <tensor> -> <LSTMStateTuple>
            c, h = tf.unstack(state_prev)
            state_prev = rnn.LSTMStateTuple(c,h)

        output, state = cell(inputs_tm[i], state_prev)
        # add state, output to list
        states = states.write(i+1, state)
        outputs = outputs.write(i, output)
        i = tf.add(i,1)
        return i, states, outputs

    with tf.variable_scope(scope):
        # initial state
        states = states.write(0, init_state)
        i = tf.constant(0)
        # stopping condition
        c = lambda x, y, z : tf.less(x, timesteps)
        # body
        b = lambda x, y, z : step(x, y, z)
        # execution 
        _, fstates, foutputs = tf.while_loop(c,b, [i, states, outputs])
        
        # if LSTM, project states
        if is_lstm(cell):
            d1 = 2*cell.state_size.c
            d2 = proj_dim if proj_dim else d1//2
            return foutputs.stack(), project_lstm_states(fstates.stack()[1:], d1, d2)

    return foutputs.stack(), fstates.stack()[1:]


def project_lstm_states(states, d1, d2):
    shapes = tf.unstack(tf.shape(states))
    batch_size = shapes[2]
    # [time_steps, 2, batch_size, d] -> [batch_size, 2, time_steps, d]
    states = tf.transpose(states, [2,0,1,3])
    # [?, 2*d] -> [?, d]
    states = tf.contrib.layers.linear(tf.reshape(states, [-1, d1]), num_outputs=d2)
    # [time_steps, batch_size, d]
    return tf.transpose(tf.reshape(states, [batch_size, -1, d2 ]), [1,0,2])


def is_lstm(cell):
    # infer from state_size
    return not isinstance(cell.state_size, int)


'''
    Bi-directional RNN

    [usage]
    states_f, states_b = bi_net(cell_f= gru_n(hdim,3),
                                        cell_b= gru_n(hdim,3),
                                        inputs= inputs_emb,
                                        batch_size= batch_size,
                                        timesteps=L,
                                        scope='bi_net_5')
'''
def bi_net(cell_f, cell_b, inputs, batch_size, timesteps, scope= 'bi_net'):
    # forward
    _, states_f = uni_net_static(cell_f, 
                          inputs,
                          cell_f.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_f')
    # backward
    _, states_b = uni_net_static(cell_b, 
                          tf.reverse(inputs, axis=[1]),
                          cell_b.zero_state(batch_size, tf.float32),
                          timesteps,
                          scope=scope + '_b')
    
    return states_f, states_b


def bi_net_dynamic(cell_f, cell_b, inputs):
    # forward
    _, states_f = uni_net_dynamic(cell_f, inputs, scope='forward')
    # backward
    _, states_b = uni_net_dynamic(cell_b, tf.reverse(inputs, axis=[1]), 
                                  scope='backward')

    return (states_f, states_b)


'''
    Attentive Decoder

    [usage]
    dec_outputs, dec_states = attentive_decoder(enc_states,
                                    tf.zeros(dtype=tf.float32, shape=[B,d]),
                                    batch_size=B,timesteps=L,feed_previous=True,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]


'''
def attentive_decoder(enc_states, init_state, batch_size, 
                      d, timesteps,
                      inputs = [],
                      scope='attentive_decoder_0',
                      num_layers=1,
                      feed_previous=False):
    # get parameters
    U,W,C,Ur,Wr,Cr,Uz,Wz,Cz,Uo,Vo,Co = get_variables(12, [d,d], name='decoder_param')
    Wa, Ua = get_variables(2, [d,d], 'att')
    Va = tf.get_variable('Va', shape=[d, 1], dtype=tf.float32)
    att_params = {
        'Wa' : Wa, 'Ua' : Ua, 'Va' : Va
    }
    
        
    def step(input_, state, ci):
        z = tf.nn.sigmoid(tf.matmul(input_, Wz)+tf.matmul(state, Uz)+tf.matmul(ci, Cz))
        r = tf.nn.sigmoid(tf.matmul(input_, Wr)+tf.matmul(state, Ur)+tf.matmul(ci, Cr))
        si = tf.nn.tanh(tf.matmul(input_, W)+tf.matmul(ci, C)+tf.matmul(r*state, U))
        
        state = (1-z)*state + z*si
        output = tf.matmul(state, Uo) + tf.matmul(input_, Vo) + tf.matmul(ci, Co)
        
        return output, state
    
    outputs = [inputs[0]] # include GO token as init input
    states = [init_state]
    for i in range(timesteps):
        input_ = outputs[-1] if feed_previous else inputs[i]
        output, state = step(input_, states[-1],
                            attention(enc_states, states[-1], att_params, d, timesteps))
    
        outputs.append(output)
        states.append(state)
    # time major -> batch major
    states_bm = tf.transpose(tf.stack(states[1:]), [1, 0, 2])
    outputs_bm = tf.transpose(tf.stack(outputs[1:]), [1, 0, 2])
    return outputs_bm, states_bm
