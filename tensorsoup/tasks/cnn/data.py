import numpy as np
import random

import sys
sys.path.append('../../')

TRAIN, TEST, VALID = 'training', 'test', 'validation'
DSETS = TRAIN, TEST, VALID

from proc import load_data, buildDictionary
from datafeed import DataFeed


import logging
from logging.config import dictConfig

logging_config = dict(
    version = 1,
    formatters = {
        'f': {'format':
              '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
    handlers = {
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.DEBUG}
        },
    root = {
        'handlers': ['h'],
        'level': logging.DEBUG,
        },
)

dictConfig(logging_config)

log = logging.getLogger('tasks.cnn.proc')

print('info>> logging works')

from pprint import pprint, pformat
import pickle

class DataLoader(object):
    '''
    loads data and builds vocabs
    creates DataFeed objects for training, test and validation datasets
    DataFeed expects a dictionary of dataset-items 
       - dformat(['contexts', 'questions', 'answers'])
    '''
    
    def __init__(self,
                 rootdir = '../../../datasets/cnn',
                 dsets   = ['training', 'test'],
                 dformat = ['contexts', 'questions', 'candidates', 'answers'],
                 nsamples = 10000,
                 max_len = 30,
                 windows = None,
                 
                 pickle_filename = None  
    ):

        if pickle_filename:
            self.__dict__ = pickle.load(open(pickle_filename, 'rb'))
            for dset in self.vdata.keys():
                for field in self.vdata[dset].keys():
                    print('DataLoader: {}.{} len --> {}'.format(dset, field, len(self.vdata[dset][field])))

            return
        
        self.rootdir = rootdir
        self.dformat = dformat
        self.max_len = max_len
        
        self.data = {}
        for dset in dsets:
            self.data[dset] = load_data(rootdir, dset, dformat=dformat, nsamples=nsamples)
            
        self.build_vocab()
        self.build_vectors()

    def memnet_compatible_metadata(self):
        return {
            'slen' : max([len(s) for s in self.data['contexts']]),
            'qlen' : max([len(q) for q in self.data['questions']]),
            'w2i'  : self.vocab['global'].word2idx,
            'i2w'  : self.vocab['global'].idx2word,
            'vocab_size' : self.vocab['global'].size,
            'clen' : min(max([len(c) for c in data['contexts']]), MAX_MEMORY_SIZE), #TODO: @suriya??
            'special_tokens' : self.initial_vocab,
            'candidates' : {
                'w2i'  : self.vocab['candidates'].word2idx,
                'i2w'  : self.vocab['candidates'].idx2word,
                'vocab_size' : self.vocab['candidates'].size,
                }
            }
    
    def build_vocab(self, vocabs = ['contexts', 'answers', 'candidates', 'questions']):
        # Build vocabulary for each fields and also a global one
        self.vocab = {'global' : buildDictionary(['PAD', 'UNK', '<eos>'])}
        global_vocab = []
        for dkey, dval in self.data.items():

            for field in dval.keys():
                print('info>> building vocabulary for {}.{}'.format(dkey, field))                
                if field not in vocabs:
                    continue
                
                if field in self.vocab.keys():
                    self.vocab[field] += buildDictionary(['PAD', 'UNK', '<eos>'],
                                                         self.max_len, dval[field])
                else:
                    self.vocab[field]  = buildDictionary(['PAD', 'UNK', '<eos>'],
                                                          self.max_len, dval[field])

                global_vocab += self.vocab[field].idx2word
                global_vocab = list(set(global_vocab))
        print('building global vocabulary')
        self.vocab['global'] = buildDictionary(['PAD', 'UNK', '<eos>'],
                                               self.max_len, global_vocab)
        

    def build_vectors(self, vocab_pairs=[( ('contexts', 'questions', 'candidates'), 'global' ),
                                         ( ('answers',)                            , 'answers')
    ]):

        self.vocab_pairs = {}
        for a, b in vocab_pairs:
            
            self.vocab_pairs.update( { i:b for i in a } )
            
        print(self.vocab_pairs)
        self.vdata = {}
        for dset in self.data.keys():
            self.vdata[dset] = {}
            
            for field in set(self.data[dset].keys()) & set(self.vocab_pairs.keys()):
                corr_vocab = self.vocab_pairs[field]
                print('info>> building vectors for {}.{} from {}'.format(dset, field, corr_vocab))                
                print(field, corr_vocab)
                self.vdata[dset][field] = vectorize_tree(self.data[dset][field],
                                                         self.vocab[corr_vocab])
            
    def __call__(self, dset='training'):
        return DataFeed(self.dformat, self.vdata[dset])

    def lookup(self, ids, vocab='test'):
        return [self.vocab[vocab].idx2word[i] for i in ids]


    def pickle(self, filename):
        pickle.dump(self.__dict__, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)

    
def vectorize_tree(node, vocab):

    if type(node) == type([]):
        return [ vectorize_tree(u, vocab) for u in node ]
    elif type(node) == type('string'):
        if node in vocab.word2idx:
            return vocab.word2idx[node]
        else :
            return vocab.word2idx['UNK']
        
if __name__ == '__main__':


    def sanity():
        print(dir(loader))
        print( loader.data[TEST]['questions'][0], loader.lookup(loader.vdata[TEST]['questions'][0], 'global') )

        for dset in loader.vdata.keys():
            #answer - single word
            print('verifying {}'.format(dset))
            for answer in loader.vdata[dset]['answers']:
                assert len(answer) == 1, 'answer "{}" not single token'.format(answer)

            print('info>> all answers are single word')
            #num of examples match across data and vdata
            for field in loader.vdata[dset].keys():
                print('{}.{} len --> {}'.format(dset, field, len(loader.vdata[dset][field])))
                assert len(loader.data[dset][field]) == len(loader.vdata[dset][field]), 'failed: num of samples not equal {} {} -- {} != {}'.format(dset, field,
                                                                                                                                                len(loader.data[dset][field]),
                                                                                                                                                len(loader.vdata[dset][field]))
    """
            print('info>> number of samples in data and vdata are equal')
            #all words are in global vocab
            for field in loader.data[dset].keys():
                for sample in loader.data[dset][field]:
                    for word in sample:
                        if len(word) > loader.max_len:
                            continue
                        assert word in loader.vocab['global'].idx2word,'{} from {}.{}not in global vocab'.format(word, dset, field)
            print('info>> all the words are in global vocab')
    """
    
    loader = DataLoader(dsets=[TEST, TRAIN], nsamples=10000)
    sanity()
    loader.pickle(loader.rootdir+'/dump01.pkl')

    train_feed = loader()
    c1, q, c2, a = train_feed.next_batch(batch_size=10)
    print('##############################')
    print(len(c1), c1)
    print('##############################')
    print(len(q), q)
    print('##############################')
    print(len(c2), c2)
    print('##############################')
    print(len(a), a)


    loader = DataLoader(pickle_filename='../../../datasets/cnn/dump01.pkl')
    train_feed = loader()
    c1, q, c2, a = train_feed.next_batch(batch_size=10)
    print('##############################')
    print(len(c1), c1)
    print('##############################')
    print(len(q), q)
    print('##############################')
    print(len(c2), c2)
    print('##############################')
    print(len(a), a)

