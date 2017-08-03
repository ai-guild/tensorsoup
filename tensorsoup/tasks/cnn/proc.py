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


def fetch_data(tag, path, window_size, 
        max_windows, qlen):

    # get list of files
    files = list_of_files(path + '/' + tag)

    # init dict
    data = {
        'queries' : [],
        'answers' : [],
        'candidates' : [],
        'windows' : [],
        'window_targets' : []
    }

    # collect all useful words for vocab
    words = []

    for f in tqdm(files):
        s,q,a = extract_structured_data(f)

        # preprocess q
        q = preprocess_text(q)

        # check num words in query
        if len(q.split(' ')) > qlen:
            continue

        # preprocess story, answer
        s = preprocess_text(s)
        a = preprocess_text(a)

        # get candidates from story
        c = get_candidates(s)

        # build windows and window targets
        w, wt = story2windows(s,c,a, window_size)

        # check for max num of windows
        if len(w) > max_windows:
            continue

        # collect words
        words.append(q.split(' '))
        words.append([ word for wi in w for word in wi ])

        # add to list
        data['queries'].append(q)
        data['answers'].append(a)
        data['candidates'].append(c)
        data['windows'].append(w)
        data['window_targets'].append(wt)

    # add list of unique words to data dict
    data['words'] = list(set(words))

    return data


def get_candidates(story):
    return list(set([ w for w in story.split(' ') 
        if '@entity' in w ]))


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

    if os.path.isfile(path + '/' + tag + '.buffer'):
        print(':: <process> [1/2] {}.buffer present'.format(tag))
        print(':: <process> [2/2] Reading from buffer')
        with open(path + '/' + tag + '.buffer', 'rb') as handle:
            return pickle.load(handle)

    print(':: <process>', tag)

    # fetch data from file; preprocess
    print('::\t [1/2] Fetch data from file')
    data = fetch_data(tag, path, window_size, max_windows=80, qlen=30)

    # save processed data to file
    print('::\t [2/2] Write data to pickle')
    with open(path + '/' + tag + '.buffer', 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    return data


def generate(path, window_size, serialize=True, 
        run_tests=False):

    # integrate train, test, valid data
    data = { }

    # fetch processed data for each tag
    words = []
    for i, tag in enumerate(TAGS):
        print(':: [{}/3] Process {}'.format(i, tag))
        data[tag] = process(tag, path, window_size)
        words.extend(data[tag]['words'])

    # build vocabulary
    print(':: [1/2] Build vocabulary')
    vocab = special_tokens + sorted(list(set(words)))

    # build metadata
    print(':: [2/2] Gather metadata')
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
