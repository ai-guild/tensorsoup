import tensorflow as tf

import sys
sys.path.append('../../')

from models.matchlstm.model import MatchLSTM
from tasks.squad.data import DataSource
from train.trainer import Trainer


if __name__ == '__main__':
    d = 10 # hidden dim
    batch_size = 8
    # instantiate model
    model = MatchLSTM(emb_dim=100, hidden_dim=d, num_indices=2, lr=0.0001)
    # create data source (SQuAD)
    datasrc = DataSource(batch_size=2)
    with tf.Session() as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size)

        # fit model
        trainer.fit(epochs=10)

