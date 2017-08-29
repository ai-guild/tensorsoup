import sys
sys.path.append('../../')

ROOT= '../../../datasets/cnn/'

#corresponding directories
TRAIN, TEST, VALID = 'training', 'test', 'validation'
TAGS = TRAIN, TEST, VALID

lass Dictionary(object):
    def __init__(self, initial_vocab):
        self.word2idx = {}
        self.idx2word = []
        self.size  = 0
        self.word_counter = {}
    
        self.add_words(initial_vocab)
    
    def add_word(self, word):
        if len(word) > 30:
            print(word) 
            exit(1)
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



import os
from pprint import pprint
from tqdm import tqdm
from tproc.utils import preprocess_text

def fetch_data(rootdir, dset=TEST):
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
        Contexts.append(context)
        Questions.append(question)
        Candidates.append(candidates)
        Answers.append(answer)
        
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
    pickleSet(ROOT+'/processed_questions/test', Contexts, Questions, Candidates, Answers, Origwords)
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
def buildDictionary(*args):
    '''
    args - list (of list) of words
    '''
    dictionary = Dictionary(['PAD', '<eos>'])
    for i,  text in enumerate(args):
        print('processing {}th element'.format(i))
        text = flatten(text)
            text = flatten(text)
        dictionary.add_words(text)
            
    return dictionary
