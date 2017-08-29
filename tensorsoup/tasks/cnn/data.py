import numpy as np
import random

import sys
sys.path.append('../../')

TRAIN, TEST, VALID = 'training', 'test', 'validation'
DSETS = TRAIN, TEST, VALID

from tasks.cnn.proc import load_data, buildDictionary

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
                 dformat = ['contexts', 'questions', 'answers'],
                 windows = None
    ):
              
        self.rootdir = rootdir
        self.dformat = dformat

        self.data = {}

        for dset in dsets:
            self.data[dset] = load_data(rootdir, dset)

        self.build_vocab()
        self.build_vectors()

    def build_vocab(self, vocabs = ['contexts', 'answers', 'candidates', 'questions']):
        # Build vocabulary for each fields and also a global one
        self.vocab = {'global' : buildDictionary(['PAD', 'UNK', '<eos>'], [])}

        for dkey, dval in self.data.items():
            for field in dval.keys():
                
                if field not in vocabs:
                    continue
                
                if field in self.vocab.keys():
                    self.vocab[field] += buildDictionary(['PAD', 'UNK', '<eos>'], dval[field])
                else:
                    self.vocab[field]  = buildDictionary(['PAD', 'UNK', '<eos>'], dval[field])

                self.vocab['global'] += self.vocab[field]


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
                print(field, corr_vocab)
                self.vdata[dset][field] = vectorize_tree(self.data[dset][field],
                                                         self.vocab[corr_vocab])
            
    def __call__(self, dset='training'):
        return DataFeed(dformat, self.vdata[dset])

    def lookup(self, ids, vocab='test'):
        return [self.vocab[vocab].idx2word[i] for i in ids]
    
def vectorize_tree(node, vocab):

    if type(node) == type([]):
        return [ vectorize_tree(u, vocab) for u in node ]
    elif type(node) == type('string'):
        return vocab.word2idx[node]

    
