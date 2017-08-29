import numpy as np
import random

import sys
sys.path.append('../../')

import tasks.cnn.proc as load_data

class DataLoader(object):
    '''
    loads data and builds vocabs
    creates DataFeed objects for training, test and validation datasets
    DataFeed expects a dictionary of dataset-items 
       - dformat(['contexts', 'questions', 'answers'])
    '''
    
    def __init__(self,
                 rootdir = '../../../datasets/cnn',
                 dformat = ['context', 'question', 'answer'],
                 windows = None
                 dsets   = ['training', 'test']
    ):
              
        self.rootdir = rootdir
        self.dformat = dformat

        self.data = {}

        for dset in dsets:
            self.data[dset] = load_data(rootdir, dset)

        self.build_vocab()

    def buildVocab(self, vocabs = ['context', 'answers'])
        # Build vocabulary for each fields and also a global one
        self.vocab = {'global' : buildDictionary(['PAD', 'UNK', '<eos>'], [])}
        for dkey, dval in self.data.values():
            for field in dval.keys():
                
                if field not in vocabs:
                    continue
                
                if field in self.vocab.keys():
                    self.vocab[field] += buildDictionary(['PAD', 'UNK', '<eos>'], dval[field])
                else:
                    self.vocab[field]  = buildDictionary(['PAD', 'UNK', '<eos>'], dval[field])

                self.vocab['global'] += self.vocab[field]
                
                    
    def __call__(self, dset, ):
        return DataFeed(dformat, self.data[dset])
