import numpy as np
import os
import pickle


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

        if len(seq) > maxlens[-2]:
            padded_seq = seq[:maxlens[0]]
        else:
            padded_seq = seq + [[PAD]*maxlens[-1]]*(maxlens[0]-len(seq))

        return [ pad_seq(s) for s in padded_seq ]
    
    return np.array([ pad_seq(seq) for seq in seqs ])

def list_of_files(path):
    return [ path + '/' + fname for fname in os.listdir(path) ]

def flatten(seq):
    acc = []
    def _flatten(seq):
        # check if we are at the bottom of hierarchy
        if type(seq) == str:
            return acc.append(seq)
        # we need to go deeper
        else:
            return [_flatten(item) for item in seq]

    _flatten(seq)
    return acc

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

def shuffle(data):

    def _shuffle(data_):
        # get count
        n = len(list(data_.values())[0])
        # get indices and shuffle
        indices = np.array(range(n))
        np.random.shuffle(indices) # NOTE : in-place
        # iterate through data dictionary
        #  get shuffled arrays
        return { k:data_[k][indices] for k in data_.keys() }

    #return { k:_shuffle(data[k]) for k in data.keys() }
    return _shuffle(data)

def vectorize_tree(node, dictionary):

    if type(node) == type([]):
        return [ vectorize_tree(u, dictionary) for u in node ]
    elif type(node) == type('string'):
        return dictionary.index(node)

def serialize(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def pad_seq(seqs, maxlen=0, PAD=0, truncate=False):

    # pad sequence with PAD
    #  if seqs is a list of lists
    if type(seqs[0]) == type([]):

        # get maximum length of sequence
        maxlen = maxlen if maxlen else seq_maxlen(seqs)

        def pad_seq_(seq):
            if truncate and len(seq) > maxlen:
                # truncate sequence
                return seq[:maxlen]

            # return padded
            return seq + [PAD]*(maxlen-len(seq))

        seqs = [ pad_seq_(seq) for seq in seqs ]
    
    # return numpy array
    return np.array(seqs, dtype=np.int32)

seq_maxlen = lambda seqs : max([len(seq) for seq in seqs])
