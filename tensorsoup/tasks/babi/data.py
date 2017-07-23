import numpy as np
import random
from itertools import chain
from six.moves import range, reduce
from pprint import pprint
from tasks.babi.proc import load_task, vectorize_data

class DataSourceAllTasks(object):
     
    def __init__(self, datadir, task_id=0, batch_size=128):

        self.datadir = datadir
        self.batch_size = batch_size
        self.task_id = task_id

        # current iteration
        self.i = [0]*21
        
        #if os.path.isfile(datadir + '/data.train'):
        # load data
        data_all, metadata  = self.fetch()
            
        print('data fetched for {} tasks'.format(len(data_all['trS'])))
        print('** metadata')
        pprint(metadata)
        self.metadata = metadata

        self.data = [{}]*21
        self.n = [{}]*21
        for task_id in range(21):
            self.data[task_id] = {}

            # convenience dict
            self.data[task_id]['train'] = [ data_all['trS'][task_id],
                                            data_all['trQ'][task_id],
                                            data_all['trA'][task_id] ]
            self.data[task_id]['test'] = [ data_all['teS'][task_id],
                                           data_all['teQ'][task_id],
                                           data_all['teA'][task_id] ]
            
            # num of examples
            self.n[task_id] = {}
            self.n[task_id]['train'] = len(self.data[task_id]['train'][0])
            self.n[task_id]['test']  = len(self.data[task_id]['test'][0])

            print('built data loader for task_id={} -- datalen={},{}'
                  .format(task_id, self.n[task_id]['train'], self.n[task_id]['test'] ))
            

    def getN(self, dtype='train'):
        return self.n[self.task_id][dtype]

    def setI(self, val = 0, dtype='train'):
        self.i[self.task_id]= val

    
    def batch(self, i, dtype='train'):
        # fetch 'i'th batch
        s, e = self.batch_size * i, (i+1)* self.batch_size
        return [ d[s:e] for d in self.data[self.task_id][dtype] ]

    def next_batch(self, n, dtype='train'):
        bi = self.batch(self.i[self.task_id], dtype=dtype)
        if self.i[self.task_id] < self.n[self.task_id][dtype]//self.batch_size:
            self.i[self.task_id] = self.i[self.task_id] + 1
        else:
            self.i[self.task_id] = 0
        return bi

    def fetch(self):
        
        def load_all_tasks(data_dir):
            train_data, test_data = [], []
            for i in range(1,21):
                train, test = load_task(data_dir, i)
                train_data.extend(train)
                test_data.extend(test)

            random.shuffle(train_data)
            random.shuffle(test_data)
            
            return train_data, test_data
        
        # data directory
        datadir = self.datadir #+ '/en-10k/'
        # task data
        
        train, test = load_all_tasks(datadir)
        data = train + test
        print('** data len', len(data))

        # metadata
        vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        # sizes
        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
        sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        query_size = max(map(len, (q for _, q, _ in data)))
        memory_size = min(50, max_story_size)
        vocab_size = len(word_idx) + 1 # +1 for nil word
        sentence_size = max(query_size, sentence_size) # for the position


        train_tasks, test_tasks = [{}]*21, [{}]*21
        trainS, trainQ, trainA = [{}]*21, [{}]*21, [{}]*21
        testS, testQ, testA = [{}]*21, [{}]*21, [{}]*21
        n_train, n_test = [{}]*21, [{}]*21

        task_id = 0
        # train/test sets
        trainS[task_id], trainQ[task_id], trainA[task_id] = vectorize_data(train, word_idx, sentence_size, memory_size)
        testS[task_id], testQ[task_id], testA[task_id] = vectorize_data(test, word_idx, sentence_size, memory_size)
        
        # params
        n_train[task_id] = trainS[task_id].shape[0]
        n_test[task_id] = testS[task_id].shape[0]
        
        
        for task_id in range(1, 21):
            train_tasks[task_id], test_tasks[task_id] = load_task(datadir, task_id)
        
            # train/test sets
            trainS[task_id], trainQ[task_id], trainA[task_id] = vectorize_data(train_tasks[task_id], word_idx, sentence_size, memory_size)
            testS[task_id], testQ[task_id], testA[task_id] = vectorize_data(test_tasks[task_id], word_idx, sentence_size, memory_size)
            
            # params
            n_train[task_id] = trainS[task_id].shape[0]
            n_test[task_id] = testS[task_id].shape[0]

        task_id = 0
        batches = zip(range(0, n_train[task_id]-self.batch_size, self.batch_size), 
                range(self.batch_size, n_train[task_id], self.batch_size))
        batches = [(start, end) for start, end in batches]

        data = {
            'trS' : trainS,
            'trQ' : trainQ,
            'trA' : trainA,
            'teS' : testS,
            'teQ' : testQ,
            'teA' : testA,
            'batches' : batches
            }


        metadata = {
                'vocab_size' : vocab_size,
                'vocab' : vocab,
                'word_idx' : word_idx,
                'sentence_size' : sentence_size,
                'memory_size' : memory_size
                }

        return data, metadata


class DataSource(object):
     
    def __init__(self, datadir, task_id, batch_size):

        self.datadir = datadir
        self.batch_size = batch_size
        self.task_id = task_id

        #if os.path.isfile(datadir + '/data.train'):
        # load data
        data_all, metadata  = self.fetch()
            
        self.metadata = metadata

        self.data = {}

        # convenience dict
        self.data['train'] = [ data_all['trS'], data_all['trQ'], data_all['trA'] ]
        self.data['test'] = [ data_all['teS'], data_all['teQ'], data_all['teA'] ]

        # num of examples
        self.n = {}
        self.n['train'] = len(self.data['train'][0])
        self.n['test']  = len(self.data['test'][0])

        # current iteration
        self.i = 0


    def batch(self, i, dtype='train'):
        # fetch 'i'th batch
        s, e = self.batch_size * i, (i+1)* self.batch_size
        return [ d[s:e] for d in self.data[dtype] ]

    def next_batch(self, n, dtype='train'):
        bi = self.batch(self.i, dtype=dtype)
        if self.i < self.n[dtype]//self.batch_size:
            self.i = self.i + 1
        else:
            self.i = 0
        return bi

    def fetch(self):
        
        def load_all_tasks(data_dir):
            train_data, test_data = [], []
            for i in range(1,21):
                train, test = load_task(data_dir, i)
                train_data.extend(train)
                test_data.extend(test)

            random.shuffle(train_data)
            random.shuffle(test_data)
            
            return train_data, test_data
        
        # data directory
        datadir = self.datadir #+ '/en-10k/'
        # task data
        if self.task_id > 0:
            train, test = load_task(datadir, task_id)
        else:
            train, test = load_all_tasks(datadir)
        data = train + test
        print('data len', len(data))

        # metadata
        vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        # sizes
        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
        sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        query_size = max(map(len, (q for _, q, _ in data)))
        memory_size = min(50, max_story_size)
        vocab_size = len(word_idx) + 1 # +1 for nil word
        sentence_size = max(query_size, sentence_size) # for the position

        # train/test sets
        trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size)
        testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

        # params
        n_train = trainS.shape[0]
        n_test = testS.shape[0]

        batches = zip(range(0, n_train-self.batch_size, self.batch_size), 
                range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]

        data = {
            'trS' : trainS,
            'trQ' : trainQ,
            'trA' : trainA,
            'teS' : testS,
            'teQ' : testQ,
            'teA' : testA,
            'batches' : batches
            }


        metadata = {
                'vocab_size' : vocab_size,
                'vocab' : vocab,
                'word_idx' : word_idx,
                'sentence_size' : sentence_size,
                'memory_size' : memory_size
                }

        return data, metadata
