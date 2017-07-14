import tensorflow as tf

import sys
sys.path.append('../../')

from models.matchlstm.model import MatchLSTM
from tasks.squad.data import DataSource
from train.trainer import Trainer


if __name__ == '__main__':
    d = 150 # hidden dim
    batch_size = 30
    # instantiate model
    model = MatchLSTM(emb_dim=200, hidden_dim=d, lr=0.001)
    # create data source (SQuAD)
    datasrc = DataSource(batch_size, glove_file='../../../datasets/glove/glove.6B.200d.txt')

    # gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size)

        # fit model
        trainer.fit(epochs=30)
