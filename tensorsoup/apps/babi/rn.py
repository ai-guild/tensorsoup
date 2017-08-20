import tensorflow as tf

import sys
sys.path.append('../../')

from models.rn.rn import RelationNet
from train.trainer import Trainer
from tasks.babi.proc import gather
from datafeed import DataFeed

from apps.babi.memnet import eval_joint_model


def train_separate(task, dataset='1k', iterations=1,
        batch_size=128):

    # get data for task
    data, metadata = gather(dataset, task)

    # build data format
    dformat = [ 'contexts', 'questions', 'answers' ]

    # create feeds
    trainfeed = DataFeed(dformat, data=data['train'])
    testfeed  = DataFeed(dformat, data=data['test' ])

    # instantiate model
    model = RelationNet(clen=metadata['clen'], 
            qlen=metadata['qlen'],
            slen=metadata['slen'],
            vocab_size=metadata['vocab_size'],
            num_candidates=metadata['candidates']['vocab_size'])

    # info
    print(':: <task {}> [0/2] Info')
    print(':: \t memory size : {}, #candidates : {}'.format(
        metadata['clen'], metadata['candidates']['vocab_size']))

    with tf.Session() as sess:
        # run for multiple initializations
        i, accuracy = 0, [0.]
        while accuracy[-1] < 0.95 and i < iterations:
            # init session
            sess.run(tf.global_variables_initializer())

            # create trainer
            trainer = Trainer(sess, model, trainfeed, testfeed,
                    batch_size = batch_size)

            print('\n:: <task {}> ({}) [1/1] Training'.format(task, i))
            # train
            acc = trainer.fit(epochs=1000000, 
                        eval_interval=1,
                        mode=Trainer.TRAIN, 
                        verbose=True,
                        lr=0.0002)

            print(':: \tAccuracy after training: ', acc)

            # next iteration
            i += 1
            # add accuracy to list
            accuracy.append(acc)
            print(acc)

        print(':: [x/x] End of training')
        print(':: Max accuracy :', max(accuracy))

        # return model and model params
        return model, sess.run(tf.trainable_variables())


if __name__ == '__main__':
    train_separate(task=0, dataset='10k', batch_size=256)
