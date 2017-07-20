import tensorflow as tf
import sys

sys.path.append('../../')

from models.matchlstm.model import MatchLSTM
from tasks.squad.data import DataSource
from train.trainer import Trainer

from multigpu import make_parallel
from visualizer import Visualizer


if __name__ == '__main__':
    
    batch_size = 90

    # instantiate model
    model = MatchLSTM(emb_dim=300, hidden_dim=200, lr=0.0005)

    # make 'n' copies of model for data parallelism
    make_parallel(model, num_copies=4, num_gpus=4)

    # setup visualizer
    #  by default, writes to ./log/
    vis = Visualizer(interval=50)
    vis.attach_scalars(model)
    vis.attach_params() # histograms of trainable variables


    # create data source (SQuAD)
    datasrc = DataSource(batch_size, 
            glove_file='../../../datasets/glove/glove.6B.300d.txt', 
            random_x=0.2)

    # gpu config
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init session
        sess.run(tf.global_variables_initializer())

        vis.attach_graph(sess.graph)

        # init trainer
        trainer = Trainer(sess, model, datasrc, batch_size, rand=True)

        # fit model
        trainer.fit(epochs=1000, visualizer=vis)
