import tensorflow as tf
import sys

sys.path.append('../../')

from models.matchlstm.model_multi import MatchLSTM
from tasks.squad.data import DataSource
from train.trainer_multi import Trainer



if __name__ == '__main__':
    
    batch_size = 30

    # instantiate model
    model = MatchLSTM(emb_dim=300, hidden_dim=200)

    # make 'n' copies of model for data parallelism
    model.make_parallel(num_copies=8, num_gpus=4)

    # create data source (SQuAD)
    datasrc = DataSource(batch_size, glove_file='../../../datasets/glove/glove.6B.300d.txt')

    # gpu config
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size)

        # fit model
        trainer.fit(epochs=1000)
