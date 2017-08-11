import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.memn2n import MemoryNet
from train.trainer import Trainer
from tasks.babi.proc import gather
from datafeed import DataFeed
from graph import *
from visualizer import Visualizer


def train_separate(task, dataset='1k', iterations=1,
        batch_size=32):

    # get data for task
    data, metadata = gather(dataset, task)

    # build data format
    dformat = [ 'contexts', 'questions', 'answers' ]

    # create feeds
    trainfeed = DataFeed(dformat, data=data['train'])
    testfeed  = DataFeed(dformat, data=data['test' ])

    hdim = 20 if task else 50
    eval_interval = 100 if task else 10
    batch_size = 32 if task else 128

    # instantiate model
    model = MemoryNet(hdim=20, num_hops=3, 
             memsize=metadata['clen'],
             sentence_size=metadata['slen'],
             qlen=metadata['qlen'],
             vocab_size=metadata['vocab_size'],
             num_candidates = metadata['candidates']['vocab_size'])

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

            print('\n:: <task {}> ({}) [1/2] Pretraining'.format(task, i))
            # pretrain
            acc = trainer.fit(epochs=100000, 
                    eval_interval=1,
                    mode=Trainer.PRETRAIN, 
                    verbose=False,
                    lr=0.0005)

            print(':: \tAccuracy after pretraining: ', acc)
            
            print('\n:: <task {}> ({}) [2/2] Training'.format(task, i))
            # train
            acc = trainer.fit(epochs=1000000, 
                        eval_interval=eval_interval,
                        mode=Trainer.TRAIN, 
                        verbose=False,
                        lr=0.0005)

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


def train_separate_all(dataset='1k'):

    batch_size = 64

    task_max_acc = []
    for task in range(1,21):
        # get task 1
        #task = 18
        data, metadata = gather('1k', task)

        # gather info from metadata
        num_candidates = metadata['candidates']['vocab_size']
        vocab_size = metadata['vocab_size']
        memsize = metadata['clen']
        sentence_size = metadata['slen']
        qlen = metadata['qlen']

        print(':: <task {}> memory size : {}'.format(task, memsize))

        # build data format
        dformat = [ 'contexts', 'questions', 'answers' ]

        # create feeds
        trainfeed = DataFeed(dformat, data=data['train'])
        testfeed  = DataFeed(dformat, data=data['test' ])

        # instantiate model
        model = MemoryNet(hdim=20, num_hops=3, memsize=memsize,
                          sentence_size=sentence_size,
                          qlen=qlen,
                          vocab_size=vocab_size,
                          num_candidates = num_candidates)

        with tf.Session() as sess:
            # run for multiple initializations
            i, accuracy = 0, [0.]
            while accuracy[-1] < 0.95 and i < 5:
                # init session
                sess.run(tf.global_variables_initializer())

                # create trainer
                trainer = Trainer(sess, model, trainfeed, testfeed,
                        batch_size = batch_size)

                print('\n:: <task {}> ({}) [1/2] Pretraining'.format(task, i))
                # pretrain
                acc = trainer.fit(epochs=100000, eval_interval=1,
                            mode=Trainer.PRETRAIN, verbose=False,
                            batch_size=64, lr=0.0005)
                print(':: \tAccuracy after pretraining: ', acc)
                
                print('\n:: <task {}> ({}) [2/2] Training'.format(task, i))
                # train
                acc = trainer.fit(epochs=1000000, eval_interval=10,
                            mode=Trainer.TRAIN, verbose=False,
                            batch_size=64, lr=0.0005)
                print(':: \tAccuracy after training: ', acc)

                # next iteration
                i += 1
                # add accuracy to list
                accuracy.append(acc)
                print(acc)

            print('Experiment Results : ')
            for i, a in enumerate(accuracy[1:]):
                print(i, a)

        task_max_acc.append(max(accuracy))

    print('____________________________________________')
    for i, acc in enumerate(task_max_acc):
        print('Task ({}) : {}'.format(i+1, acc))
    print('____________________________________________')



if __name__ == '__main__':
    # train joint model
    model, model_params = train_separate(task=0, dataset='10k')

    # evaluate on individual tasks
    eval_joint_model(model, model_params)
