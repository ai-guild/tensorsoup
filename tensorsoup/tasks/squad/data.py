import json
import pickle
import os

import numpy as np
import random

from nltk import word_tokenize


class DataSource(object):

    def __init__(self, batch_size, datadir='../../../datasets/SQuAD/',
            glove_file='../../../datasets/glove/glove.6B.100d.txt'):

        # 100d -> glove_file='datasets/glove/glove.6B.100d.txt'):
        # 200d -> glove_file='datasets/glove/glove.6B.200d.txt'):
         
        self.datadir = datadir
        self.batch_size = batch_size

        # check if processed data exists
        if os.path.isfile(datadir + '/data.train'):
            # load data
            self.train = self.load(datadir, tag='train')
            self.test   = self.load(datadir, tag='test')
        else:
            # prepare data
            self.process(datadir)

        # convenience dict
        self.data = { 'train' : self.train, 'test' : self.test }

        # data cache
        self.cache = { 'train' : {}, 'test' : {} }

        # num of examples
        self.n = {}
        self.n['train'] = len(self.data['train']['passages'])
        self.n['test']   = len(self.data['test']['passages'])

        print('Initializing Glove Model ...')
        self.glove = self.loadGloveModel(glove_file)

        # infer glove dimensions
        self.glove_dim = len(self.glove['ocelot'])

        # current iteration
        self.i = 0

        # list of batch id's
        #  to sample from
        self.batches = {}
        self.batches['train'] = list(range(self.n['train']))
        self.batches['test'] = list(range(self.n['test']))


    def process(self, datadir='datasets/SQuAD/'):
        # prepare train set
        print('Preparing Training Set ...')

        def shuffle(a,b,c,d):
            idx = list(range(len(a)))
            random.shuffle(idx)
            sa, sb, sc, sd = [], [], [], []
            for i in idx:
                sa.append(a[i])
                sb.append(b[i])
                sc.append(c[i])
                sd.append(d[i])
            return sa, sb, sc, sd

        cons,qs,sps,eps = self.get_dataset(datadir + 'train-v1.1.json')
        cons,qs,sps, eps = shuffle(cons,qs,sps, eps) 
        self.train = {}
        self.train['passages'] = cons
        self.train['queries'] = qs
        self.train['sps'] = sps
        self.train['eps'] = eps

        # prepare test set
        print('Preparing test Set ...')
        cons,qs,sps,eps = self.get_dataset(datadir + 'dev-v1.1.json')
        cons,qs,sps, eps = shuffle(cons,qs,sps, eps) 
        self.test = {}
        self.test['passages'] = cons
        self.test['queries'] = qs
        self.test['sps'] = sps
        self.test['eps'] = eps

        # save processed data
        self.save(self.train, datadir)
        self.save(self.test, datadir, tag='test')


    def save(self, data_dict, datadir='.', tag='train'):
        filename = datadir + '/data.' + tag
        print('Writing processed data to ' + filename)
        # write to disk
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)


    def load(self, datadir='.', tag='train'):
        filename = datadir + '/data.' + tag
        print('Reading processed data from ' + filename)
        # read from disk
        with open(filename, 'rb') as f:
            proc_data = pickle.load(f)

        return proc_data


    def batch(self, dtype, i, batch_size):

        # check in cache
        if i in self.cache[dtype]:
            return self.cache[dtype][i]

        # fetch 'i'th batch
        s, e = i*batch_size, (i+1)*batch_size
        p, q = self.data[dtype]['passages'][s:e], self.data[dtype]['queries'][s:e]

        # prepare padding mask
        lens = [len(item) for item in p]

        mask = np.ones([batch_size, max(lens)])

        for i,l in enumerate(lens):
            mask[i][l:] = 0.
        # tile mask to get shape 2xBxLp
        mask = np.array([mask, mask], dtype=np.float32)

        # embed passages and queries
        batch_p = self.embed_sequences(p, as_array=True)
        batch_q = self.embed_sequences(q, as_array=True)

        batch_targets = np.array([self.data[dtype]['sps'][s:e], self.data[dtype]['eps'][s:e]])

        batch_i = (batch_p, batch_q, batch_targets, mask)

        # save to cache
        self.cache[dtype][i] = batch_i

        return batch_i


    def rand_next_batch(self, dtype='train'):
        # select train/test
        i = np.random.randint(0, self.n[dtype]//self.batch_size -1)
        return self.batch(dtype, i, self.batch_size)


    def next_batch(self, dtype='train'):
        bi = self.batch(dtype, self.i, batch_size=self.batch_size)
        if self.i < self.n[dtype]//self.batch_size:
            self.i = self.i + 1
        else:
            self.i = 0
        return bi


    def next_n_batches(self, n, dtype='train'):
        bi_n = []
        for _ in range(n):
            bi_n.append(self.batch(dtype, self.i, batch_size=self.batch_size))
            if self.i < self.n[dtype]//self.batch_size:
                self.i = self.i + 1
            else:
                self.i = 0
        return bi_n


    def next_n_random_batches(self, n, dtype='train'):
        batches = random.sample(self.batches)
        bi_n = []
        for _ in range(n):
            bi_n.append(self.batch(dtype, self.i, batch_size=self.batch_size))
            if self.i < self.n[dtype]//self.batch_size:
                self.i = self.i + 1
            else:
                self.i = 0
        return bi_n


    def get_dataset(self, path):
        contexts = []
        questions = []
        start_char_positions = []
        end_char_positions = []
        start_word_positions = []
        end_word_positons = []
        with open(path) as f:
            x = json.load(f)
        for data in x['data']:
            for para in data['paragraphs']:
                for qa in para['qas']:
                    temp_ans = [] # To avoid duplicate questions with same answers
                    for answer in qa['answers']:
                        if answer['text'] not in temp_ans:
                            # fetch row
                            context = para['context']
                            question = qa['question']
                            sch_pos = answer['answer_start']
                            ech_pos = answer['answer_start'] + len((answer['text']))
                            
                            # infer answer word-level positions
                            sw_pos = len(word_tokenize(context[:sch_pos]))
                            ew_pos = len(word_tokenize(context[:ech_pos])) -1

                            contexts.append(word_tokenize(context))
                            questions.append(word_tokenize(question))
                            start_word_positions.append(sw_pos)
                            end_word_positons.append(ew_pos)

        return contexts, questions, start_word_positions, end_word_positons


    def loadGloveModel(self, gloveFile):
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model


    def embed_word(self, word):
        return np.array(self.glove[word]) if word in self.glove else np.zeros(self.glove_dim)


    def embed_sequences(self, seqs, as_array=False):
        slen, maxlen = len(seqs), max([len(i) for i in seqs])
        eseqs = []
        for seq in seqs:
            words = [self.embed_word(w) for w in seq]
            padding = [np.zeros(self.glove_dim) for i in range(maxlen - len(seq))]
            eseqs.append(words + padding)
        return np.array(eseqs) if as_array else eseqs
