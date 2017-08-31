import logging
log = logging.getLogger('tasks.cnn.proc')
log.setLevel(logging.DEBUG)
from pprint import pprint, pformat
from tqdm import tqdm
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
        try:
            if word not in self.idx2word:
                self.idx2word.append(word)
                self.word2idx[word] = self.size
                self.size += 1

            if word not in self.word_counter:
                self.word_counter[word] = 1
            else:
                self.word_counter[word] += 1
        except:
            self.log.debug('cannot add this item to dictionary" {}'.format(pformat(word)))
    
    def add_words(self, words):
        for word in tqdm(words):
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

from tproc.utils import flatten
def buildDictionary(intial_vocab=[], max_len = 30, *args):
    '''
    args - list (of list) of words
    '''

    dictionary = Dictionary(intial_vocab, max_len=max_len)
    for i,  text in enumerate(args):
        print('buildDictionary: processing {}th element'.format(i))
        text = flatten(text)
        text = [ t.split() for t in text ]
        text = set(flatten(text))
        dictionary.add_words(list(text))
            
    return dictionary

