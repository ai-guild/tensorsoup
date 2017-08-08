import numpy as np
import os


def is_word(w):
    '''
        is_word(str : w) -> Boolean
            if a word contains 
             at least 1 alphanumeric char
    '''
    return len([ch for ch in w if int(ch.isalnum()) ]) >  0

def preprocess_text(p, chars='_.,!#$%^&*()\?:{}[]/;`~*+|'):
    filtered_text = ''
    for ch in p:
        if ch not in chars:
            filtered_text = filtered_text + ch
    filtered_text = filtered_text.replace('  ', ' ').strip()
    
    allowed_words = []
    for w in filtered_text.split(' '):
        if '-' in w or '\'' in w or '"' in w:
            if is_word(w):
                allowed_words.append(w.lower())
        else:
            allowed_words.append(w.lower())
            
    return ' '.join(allowed_words)

def build_vocabulary(texts, special_tokens):

    # combine windows and queries
    #  into a text blob
    text = ' '.join(texts)
    
    # get all words
    words = text.split(' ')

    # get unique words -> vocab
    return special_tokens + sorted(list(set(words)))

def words2indices(words, metadata):
    w2i = metadata['w2i']
    return [ w2i[w] if w in w2i else w2i['unk'] for w in words ]

def indices2words(indices, metadata):
    i2w, w2i = metadata['i2w'], metadata['w2i']
    return ' '.join([ i2w[i] for i in indices if w in w2i ])

def filter_by(idata, cond):
    data = {}
    for k,v in idata.items():
        data[k] = []
        for i, item in enumerate(idata[k]):
            if cond(idata, i):
                data[k].append(item)
    return data

def pad_sequences(seqs, maxlens, metadata):

    PAD, w2i = metadata['special_tokens'][0], metadata['w2i']
    PAD = w2i[PAD]
    
    def pad_seq(seq):
        if type(seq[0]) is not list:
            return seq + [PAD]*(maxlens[-1]-len(seq))

        padded_seq = seq + [[PAD]*maxlens[-1]]*(maxlens[-2]-len(seq))
        return [ pad_seq(s) for s in padded_seq ]
    
    return np.array([ pad_seq(seq) for seq in seqs ])

def list_of_files(path):
    return [ path + '/' + fname for fname in os.listdir(path) ]

# index sequence with any number of levels
def index_seq(seq, w2i):
    
    def index_sentence(seq):
        # check if we are at the bottom of hierarchy
        if type(seq) == str:
            # get words
            words = seq.split(' ')
            # check if sequence is just a word
            if len(words) == 1:
                w = words[0]
                return w2i[w] if w in w2i else w2i['unk']
            return [w2i[w] for w in words]
        # we need to go deeper
        else:
            return [ index_sentence(item) for item in seq ]
    
    return index_sentence(seq)
