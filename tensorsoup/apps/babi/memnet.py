import tensorflow as tf

import sys
sys.path.append('../../')

from models.memorynet.memn2n import MemoryNet
from train.trainer import Trainer
from tasks.babi.proc import gather
from datafeed import DataFeed

from visualizer import Visualizer


if __name__ == '__main__':

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
                acc = trainer.fit(epochs=1000000, eval_interval=100,
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
