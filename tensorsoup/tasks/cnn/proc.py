import sys
import logging 
sys.path.append('../../')

ROOT= '../../../datasets/cnn/'

#corresponding directories
TRAIN, TEST, VALID = 'train', 'test', 'valid'
DSETS = TRAIN, TEST, VALID

import logging
log = logging.getLogger('tasks.cnn.proc')
log.setLevel(logging.DEBUG)
from pprint import pprint, pformat

import os
from pprint import pprint
from tqdm import tqdm
from tproc.utils import preprocess_text, serialize, vectorize_tree, pad_seq
from tproc.dictionary import Dictionary, buildDictionary


class CNNDict(Dictionary):
    def __init__(self, initial_vocab,
                 name = None,
                 max_len=30, logger=None):
        super(CNNDict, self).__init__(initial_vocab= initial_vocab,
                                      name=name, max_len=max_len, logger=logger)
        self.metadata = {}
        
    def add_context(self, context):
        self.metadata['clen'] = max(self.metadata['clen'], len(context))
        self.add_words(context)

    def add_question(self, question):
        self.metadata['qlen'] = max(self.metadata['qlen'], len(question))
        self.add_words(question)

    def add_answer(self, answer):
        self.add_words(answer)

    def add_candidates(self, candidates):
        self.add_words(candidates)
        
def string_to_sample(string):
    lines = string.splitlines()
    url, _, context, _, question, _, answer, _, *__candidates = lines
    candidates = []
    origwords = []
    for c in __candidates:
        candidate, origword = c.split(':', 1)
        candidates.append(candidate)
        origwords.append(origword)
        
    context, question, answer = [i.split() for i in 
                                 [context, question, answer]]
        
    return context, question, answer, candidates

def process_sample(filename, data, vocab):
    with open(filename, 'r') as sample:
        context, question, answer, candidates = string_to_sample(sample.read())

        vocab.add_context(context)
        vocab.add_question(question)
        vocab.add_answer(answer)
        vocab.add_candidates(candidates)

    data['context']    .append( vectorize_tree(context, vocab)    ) 
    data['question']   .append( vectorize_tree(question, vocab)   )
    data['answer']     .append( vectorize_tree(answer, vocab)     )
    data['candidates'] .append( vectorize_tree(candidates, vocab) )
        
    return data, vocab
    
def process_set(dirname, vocab):

    data = {'context': [], 'question' : [], 'answer':[], 'candidates':[]}
    for filename in tqdm(os.listdir(dirname)):
        data, vocab = process_sample(dirname+'/'+filename, data, vocab)

    return data, vocab

def process():

    vocab = CNNDict(['PAD', 'UNK'])
    vocab.metadata = {'clen' : 0, 'qlen':0}
    
    basedirs = {
        #'train' :  '../../../datasets/cnn/questions/training',
        'test' :  '../../../datasets/cnn/questions/test',
        #'valid' :  '../../../datasets/cnn/questions/validation',
        }

    data = {}
    for set in basedirs.keys():
        d, v = process_set(basedirs[set], vocab)
        data[set] = d

    serialize(data, ROOT + 'data.pkl')
    serialize(vocab.__dict__, ROOT + 'vocab.pkl')
    
    return data, vocab
        
def pad_data(data, vocab, truncate=False):

    metadata  = vocab.metadata
    clen = metadata['clen']
    qlen = metadata['qlen']

    padded_data = {}

    # for [train, test, valid]
    for dset in ['test']:
        # pad each field
        #padded_data[dset] = { k: pad_seq(v) for k,v in data[dset].items() }
        padded_data[dset] = {
                'context' : pad_seq(data[dset]['context'], clen, 
                    truncate=True),
                'question' : pad_seq(data[dset]['question'], qlen, 
                    truncate=True),
                'answer' : pad_seq(data[dset]['answer'], 1),
                'candidates' : pad_seq(data[dset]['candidates'], 
                    10, truncate=True)
                }
        
    return padded_data


import pickle
def gather(fresh=False):
    vocab  = CNNDict([])
    if fresh or  not os.path.exists(ROOT+'/data.pkl'):
        process()

    data = pickle.load(open(ROOT+'/data.pkl', 'rb'))
    vocab.__dict__ = pickle.load(open(ROOT+'/vocab.pkl', 'rb'))

    print(vocab.size)
    return pad_data(data, vocab), vocab


if __name__ == '__main__':

    gather(1)
