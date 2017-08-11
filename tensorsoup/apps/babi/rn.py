import tensorflow as tf

import sys
sys.path.append('../../')

from models.rn.rn import RelationNet
from train.trainer import Trainer
from tasks.babi.proc import gather
from datafeed import DataFeed
from graph import *

from visualizer import Visualizer


def train_separate(task, dataset='1k', iterations=1,
        batch_size=256):

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

    # create visualizer
    vis = Visualizer()
    vis.attach_scalars(model)

    with tf.Session() as sess:
        # run for multiple initializations
        i, accuracy, model_params  = 0, [0.], [None]
        while accuracy[-1] < 0.95 and i < iterations:
            # init session
            sess.run(tf.global_variables_initializer())

            # add graph to visualizer
            vis.attach_graph(sess.graph)

            # create trainer
            trainer = Trainer(sess, model, trainfeed, testfeed,
                    batch_size = batch_size)

            print('\n:: <task {}> ({}) [1/1] Training'.format(task, i))
            # train
            acc = trainer.fit(epochs=1000000, 
                        eval_interval=4,
                        mode=Trainer.TRAIN, 
                        verbose=True,
                        lr=2e-4,
                        visualizer=vis)

            print(':: \tAccuracy after training: ', acc)

            # next iteration
            i += 1
            # add accuracy to list
            accuracy.append(acc)
            model_params.append(sess.run(tf.trainable_variables()))
            print(acc)

        print(':: [x/x] End of training')
        print(':: Max accuracy :', max(accuracy))

        # return model and best model params
        return model, model_params[accuracy.index(max(accuracy))]


'''
    evaluate trained model (jointly trained)
        on individual tasks

'''
def eval_joint_model(model, model_params):
    # get test data for evaluation
    data, metadata = gather('10k', 0)

    # build data format
    dformat = [ 'contexts', 'questions', 'answers' ]
    
   
    accuracies = []
    with tf.Session() as sess:

        # reload model params
        sess.run(set_op(model_params))

        # build trainer
        trainer = Trainer(sess, model, batch_size=128)
 
        for i in range(1,21):
            # test feed for task 'i'
            testfeed  = DataFeed(dformat, data=data['test'][i])

            # evaluate
            loss, acc = trainer.evaluate(feed=testfeed)

            # note down task accuracy
            accuracies.append(acc)

    print('\n:: Evaluation on individual tasks\n::   Accuracy')
    for i, acc in enumerate(accuracies):
        print(':: \t[{}] {}'.format(i, acc))



if __name__ == '__main__':
    # train joint model
    model, model_params = train_separate(task=0, dataset='10k',
            batch_size=128)

    # evaluate on individual tasks
    eval_joint_model(model, model_params)
