import numpy as np


def is_word(w):
    '''
        is_word(str : w) -> Boolean
            if a word contains 
             at least 1 alphanumeric char
    '''
    return len([ch for ch in w if int(ch.isalnum()) ]) >  0

def preprocess_text(p, chars='_.,!@#$%^&*()\?:{}[]/;`~*+|'):
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

def pad_sequences(seqs, maxlen, metadata):
    '''
        assumption

            seqs : list of list of indices (of words) of variable size
    '''
    # num of sequences
    n = len(seqs)
    # special tokens -> PAD
    #  word to index
    PAD, w2i = metadata['special_tokens'][0], metadata['w2i']
    # shape : [n, maxlen]

    # for each sequence
    pad_seq = lambda seq : seq + [w2i[PAD]]*(maxlen-len(seq))
    # return padded nd.array
    return np.array([ pad_seq(seq) 
        for seq in seqs ], dtype=np.int32).reshape(n, maxlen)
