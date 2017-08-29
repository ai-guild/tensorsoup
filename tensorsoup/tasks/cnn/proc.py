import sys
import logging 
sys.path.append('../../')

ROOT= '../../../datasets/cnn/'

#corresponding directories
TRAIN, TEST, VALID = 'training', 'test', 'validation'
DSETS = TRAIN, TEST, VALID

class Dictionary(object):
    count = 0
    def __init__(self, initial_vocab,
                 name = None,
                 max_len=30, logger=None):

        self.initial_vocab = initial_vocab
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = []
        self.size  = 0
        self.word_counter = {}
    
        self.add_words(initial_vocab)

        self.name = name if name else 'dictionary-{:000d}'.format(Dictionary.count)
        self.log = logger if logger else logging.Logger(self.name)
        self.log.debug('Dictionary {} is built successfully'.format(self.name))
        Dictionary.count += 1
                
    def add_word(self, word):
        if len(word) > self.max_len:
            self.log.warning('"{}" exceeds max_len {} - ignoring it'
                             .format(word, self.max_len))
            return
        
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.size
            self.size += 1
            
        if word not in self.word_counter:
            self.word_counter[word] = 1
        else:
            self.word_counter[word] += 1
    
    def add_words(self, words):
        for word in words:
            self.add_word(word)
        
    def word(self, idx):
        if idx < self.size:
            return self.idx2word[idx]
    
    def index(self, word):
        return self.word2idx[word]
    
    def wordCount(self, word):
        return self.word_counter[word]


    def __add__(self, other):
        new = Dictionary(self.initial_vocab + other.initial_vocab)
        new.add_words(self.idx2word)
        new.add_words(other.idx2word)
        return new

import os
from pprint import pprint
from tqdm import tqdm
from tproc.utils import preprocess_text

def process_data(rootdir, dset=TEST):
    '''
    Contexts, Questions, Candidates, Answers, Origwords = fetch_data(ROOT)
    where ROOT dir contains questions/training/*.question
    '''
    samples = []
    dirname = rootdir + 'questions/' + dset
    Contexts, Questions, Candidates, Answers, Origwords = [], [], [], [], []
    for filename in tqdm(os.listdir(dirname)):
        with open(dirname+'/'+filename) as sample:
            lines = sample.read().splitlines()
            url, _, context, _, question, _, answer, _, *__candidates = lines
            candidates = []
            origwords = []
            for c in __candidates:
                candidate, origword = c.split(':', 1)
                candidates.append(candidate)
                origwords.append(origword)
                
        context, question, answer = [preprocess_text(i) for i in 
                                     [context, question, answer]]
        
        Contexts  .append( context   .split() )
        Questions .append( question  .split() )
        Answers   .append( answer    .split() )
        
        Candidates.append( candidates         )
        
    print('------url------------\n', url)
    print('-------question--------------\n', question)
    print('---------candidates-----------------\n', candidates)
    print('--------origwords-----------------\n', origwords)
    print('---------answer------------------\n', answer)
    print('---------context-----------------\n', context)
    
    return Contexts, Questions, Candidates, Answers, Origwords


import pickle
def pickleSet(dirname, contexts, questions, candidates, answers, origwords):
    '''
    pickleSet(ROOT+'/processed_questions/test', 
          Contexts, Questions, Candidates, Answers, Origwords)
    '''
    names = 'contexts', 'questions', 'candidates', 'answers', 'origwords'
    data = contexts, questions, candidates, answers, origwords
    
    for name, datum in zip(names, data):
        with open(dirname+'/'+name, 'wb') as f:
            pickle.dump(datum, f,  pickle.HIGHEST_PROTOCOL)
            
def loadSet(dirname):
    '''
    loadSet(ROOT+'/processed_questions/'+tag)
    '''
    names = 'contexts', 'questions', 'candidates', 'answers', 'origwords'
    data = []
    for name in names:
        with open(dirname+'/'+name, 'rb') as f:
            data.append(pickle.load(f))
            
    return data


from tproc.utils import flatten
def buildDictionary(intial_vocab=[], *args):
    '''
    args - list (of list) of words
    '''
    dictionary = Dictionary(intial_vocab)
    for i,  text in enumerate(args):
        print('processing {}th element'.format(i))
        text = flatten(text)
        text = [ t.split() for t in text ]
        text = flatten(text)
        dictionary.add_words(text)
            
    return dictionary



import os.path
def load_data(root, dset, dformat=None):

    processed_dir = root+'/processed_questions'
    dset_dir = processed_dir + '/' + dset
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir, 0o755)
        contexts, questions, candidates, answers, origwords = process_data(root, dset)
        os.mkdir(dset_dir, 0o755)
        pickleSet(processed_dir+'/test', 
                  contexts, questions, candidates, answers, origwords)
        
    else:

        contexts, questions, candidates, answers, origwords = loadSet(dset_dir)

    data = {}
    data['contexts']    = contexts
    data['questions']  = questions
    data['candidates'] = candidates
    data['answers']    = answers
    data['origwords']  = origwords

    if dformat:
        data = { key:data[key] for key in dformat }
    
    return data

