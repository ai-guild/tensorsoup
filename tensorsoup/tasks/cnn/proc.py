import numpy as np
import pickle
import os

import sys
sys.path.append('../../')

from tproc.utils import *
from tasks.cbt.proc import *

from tqdm import tqdm


DATA_DIR = '../../../datasets/cnnqa/questions/'

# tags
TRAIN, TEST, VALID = 'training', 'test', 'validation'
TAGS = TRAIN, VALID, TEST

# special tokens
UNK = '<unk>'
PAD = '<pad>'
special_tokens = [ PAD, UNK ]


def read_from_file(filename):
    with open(filename) as f:
        return [ line for line in f.read().split('\n') if line ]

    
def extract_structured_data(filename):
    return read_from_file(filename)[1:4]


def fetch_data(tag, path):
    files = list_of_files(path + '/' + tag)
    data = {
        'stories' : [],
        'queries' : [],
        'answers' : []
    }
    for f in tqdm(files):
        s,q,a = extract_structured_data(f)
        data['stories'].append(s)
        data['queries'].append(q)
        data['answers'].append(a)
    
    return preprocess(data)


def preprocess(idata):
    return {
        'stories' : [preprocess_text(s) for s in idata['stories']],
        'queries' : [preprocess_text(q) for q in idata['queries']],
        'answers' : idata['answers']
    }


def build_candidates(data):

    def get_candidates(story):
        return list(set([ w for w in story.split(' ') 
            if '@entity' in w ]))

    data.update( {
        'candidates' : [ get_candidates(s) for s in data['stories'] ]
        })

    return data


def gather_metadata(data, vocab):
    metadata = {
            'special_tokens' : special_tokens,
            'vocab_size' : len(vocab),
            'w2i' : { w:i for i,w in enumerate(vocab) },
            'i2w' : vocab 
            }
    qlen, memory_size = 0, 0
    for k in data.keys():
        qlen = max(qlen, max([len(q.split(' ')) for q in data[k]['queries']]))
        memory_size = max(memory_size, max([len(wi) for wi in data[k]['windows']]))

    metadata.update( {
        'qlen' : qlen, 'memory_size' : memory_size
        })

    return metadata

def process(tag, path, window_size):
    print(':: <process>', tag)

    # fetch data from file; preprocess
    print('::\t [1/5] Fetch raw data')
    raw_data = fetch_data(tag, path)

    # gather candidates from stories
    print('::\t [2/5] Build candidates')
    data = build_candidates(raw_data)

    # build windows
    if window_size:
        print('::\t [3/5] Build windows')
        data = build_windows(data, window_size)

        # remove stories
        print('::\t [4/4] Remove stories')
        del data['stories']
        assert 'stories' not in data

        # filter data based on num of windows
        print('::\t [4/5] Filter data')
        data = filter_data(data, num_windows=120, qlen=30)

    return data


def generate(path, window_size, serialize=True, 
        run_tests=False):

    # integrate train, test, valid data
    data = { }

    # fetch processed data for each tag
    texts = []
    for i, tag in enumerate(TAGS):
        print(':: [{}/3] Process {}'.format(i, tag))
        data[tag] = process(tag, path, window_size)
        texts.extend(data[tag]['queries'])
        flat_windows = [ ' '.join(wi) for w in data[tag]['windows'] 
                for wi in w ]
        texts.extend(flat_windows)


    # build vocabulary
    print(':: [1/1] Build vocabulary')
    vocab = build_vocabulary(texts, special_tokens)

    print(':: [1/1] Gather metadata')
    # build metadata
    metadata = gather_metadata(data, vocab)
    # add window info in metadata
    metadata['window_size'] = window_size 

    # index data
    for i, tag in enumerate(TAGS):
        print(':: [{}/{}] Index {}'.format(i, len(TAGS), tag))
        data[tag] = index(data[tag], metadata)

    # serialize data
    if serialize:
        print(':: [1/2] Serialize data')
        with open('{}/data.pickle'.format(path), 'wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
        print(':: [2/2] Serialize metadata')
        with open('{}/metadata.pickle'.format(path), 'wb') as handle:
            pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)

    return data, metadata


def gather(path=DATA_DIR, window_size=5, gen=False):

    # build file names
    dataf = path + '/data.pickle'
    metadataf = path + '/metadata.pickle'

    if not gen:
        if os.path.isfile(dataf) and os.path.isfile(metadataf):
            print(':: <gather> [1/2] Reading from' , dataf)
            with open(dataf, 'rb') as handle:
                data = pickle.load(handle)
            print(':: <gather> [2/2] Reading from' , metadataf)
            with open(metadataf, 'rb') as handle:
                metadata = pickle.load(handle)
            return data, metadata

    return generate(path, window_size)



if __name__ == '__main__':
    data, metadata = gather(gen=True)
