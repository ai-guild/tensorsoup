import pickle
import os

from tests import *


BASE_PATH = '../../../datasets/CBTest/data'

TYPE_VERB = 'V'
TYPE_PREP = 'P'
TYPE_NOUN = 'CN'
TYPE_NE = 'NE'


def _read_sample(n, samples):
    '''
    Fetches the nth sample from the dataset

    Arguments:
    n : int, Index of the sample to be fetched
    samples : list of strings, Each item is a
              sample containing context, query,
              candidate answers and ground-truth.
    Returns:
    - context (string)
    - query (string)
    - candidates (string)
    - answer (string)
    '''

    lines = samples[n].splitlines()
    context = ''
    for line in lines[:-1]:
        context += line.lstrip('0123456789')
    query = lines[-1].split('\t')[0].lstrip('21')
    candidates = lines[-1].split('\t')[-1].replace('|', ' ')
    answer = lines[-1].split('\t')[1]
    return context, query, candidates, answer


def fetch_data(path):
    '''
    Fetches the entire dataset and splits
    it into contexts, queries, candidate answers
    and ground-truth

    Arguments:
    path : Path to the dataset file

    Returns: A tuple of
    - contexts (list of strings)
    - queries (list of strings)
    - candidates (list of strings)
    - answers (list of strings)
    '''

    with open(path) as f:
        raw_data = f.read()
        samples = raw_data.split('\n\n')[:-1]  # Last split is empty
    contexts, queries, candidates, answers = [], [], [], []
    for i in range(len(samples)):
        co, qu, ca, an = _read_sample(i, samples)
        contexts.append(co)
        queries.append(qu)
        candidates.append(ca)
        answers.append(an)

    return (contexts, queries, candidates, answers)


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


def preprocess_candidates(c):
    # check each word
    #  remove non-words
    words = [ preprocess_text(w) for w in c.split(' ') if is_word(w) ]
    
    # add empty candidates if len(words) < 10
    words = words + ['']*(10-len(words))

    # make sure len(words) == 10
    assert len(words) == 10

    return words


def preprocess(raw_dataset):

    stories, queries, candidates, answers = raw_dataset
    processed = []

    # preprocessing
    for i, data in enumerate([stories, queries, answers]):
        print(':: <preprocess> [{}/3] preprocessing text'.format(i+1))
        processed.append([ preprocess_text(item) 
            for item in data ])

    # preprocess candidates
    print(':: <preprocess> [1/1] preprocessing candidates')
    candidates = [ preprocess_candidates(c) for c in candidates ]


    dataset = { 'stories' : processed[0],
                'queries' : processed[1],
                'answers' : processed[2]
               }

    dataset['candidates'] = candidates

    print(':: <preprocess> preprocessing complete; returning dataset')
    return dataset


def build_vocabulary(texts):
    # combine stories and queries
    #  into a text blob
    text = ' '.join(texts)
    
    # get all words
    words = text.split(' ')

    # get unique words -> vocab
    return sorted(list(set(words)))


def gather_metadata(data, vocab):
    # assumption : max story/query len of training set is larger than test/valid
    #  i'm sure this will come back to bite me in the ass
    return {
            'max_story_len' : max([len(s.split(' ')) for s in data['stories']]),
            'max_query_len' : max([len(q.split(' ')) for q in data['queries']]),
            'w2i' : { w:i for i,w in enumerate(vocab) },
            'i2w' : vocab,
            'vocab_size' : len(vocab)
            }


def story2windows(story, candidates, window_size):
    # tokenize
    words = story.split(' ')
    storylen = len(words)

    # convenience
    b = window_size
    
    def get_window(i):
        start = max(i - b//2, 0)
        end = start + b
        # check if we've exceeded the limit of story
        if end >= storylen:
            # push back start
            start = start - (end - storylen + 1)
            # update end to last index
            end = storylen - 1
        return words[start:end]
        
    return [ get_window(i) for i,w in enumerate(words)
            if w in candidates ]


def build_windows(dataset, window_size):
    return [ story2windows(s,c, window_size) for s,c in 
            zip(dataset['stories'], dataset['candidates']) ]

def index(data, metadata):
    # word to index dictionary
    w2i = metadata['w2i']

    def sent2indices(sent):
        return [ w2i[w] for w in sent.split(' ') ]

    def words2indices(words):
        return [ w2i[w] for w in words ]

    indexed_data =  {
            'stories' : [ sent2indices(s) for s in data['stories'] ],
            'queries' : [ sent2indices(q) for q in data['queries'] ],
            'candidates' : [ words2indices(c) for c in data['candidates'] ],
            'answers' : [ w2i[a] for a in data['answers'] ]
            }

    if 'windows' in data:
        indexed_data['windows'] = [ words2indices(wi) for wi in data['windows'] ]

    return indexed_data


def process(tag=TYPE_NE, run_tests=True, serialize_data=True, window_size=None):
    # build file names
    train_file = 'cbtest_{}_train.txt'.format(tag)
    valid_file = 'cbtest_{}_valid_2000ex.txt'.format(tag)
    test_file = 'cbtest_{}_test_2500ex.txt'.format(tag)

    # process files
    print(':: [1/3] Fetch TRAIN data')
    train = process_file(train_file, window_size)
    print(':: [2/3] Fetch VALID data')
    valid = process_file(valid_file, window_size)
    print(':: [3/3] Fetch TEST Data')
    test  = process_file(test_file, window_size)

    texts = []
    for data in [train, test, valid]:
        texts.extend(data['stories'])
        texts.extend(data['queries'])

    # build vocabulary
    vocab = build_vocabulary(texts)

    if run_tests:
        print(':: <test> [1/1] Test vocabulary')
        for data in [train, test, valid]:
            test_vocabulary(vocab, data)

    # metadata
    metadata = gather_metadata(train, vocab)
    # add window info in metadata
    metadata['window_size'] = window_size 

    print(':: [1/1] Index data')
    data = { 'train' : index(train, metadata),
             'test'  : index(test,  metadata),
             'valid' : index(valid, metadata)
             }

    # serialize data
    if serialize_data:
        print(':: [1/1] Serialize data and metadata')
        with open('{}/data.{}'.format(BASE_PATH, tag), 'wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
        with open('{}/metadata.{}'.format(BASE_PATH, tag), 'wb') as handle:
            pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)

    return data, metadata


def gather(tag=TYPE_NE, window_size=None):
    # build file names
    dataf = '{}/data.{}'.format(BASE_PATH, tag)
    metadataf = '{}/metadata.{}'.format(BASE_PATH, tag)

    # if processed files exist
    #  read pickle and return
    if os.path.isfile(dataf) and os.path.isfile(metadataf):
        print(':: <gather> [1/2] Reading from' , dataf)
        with open(dataf, 'rb') as handle:
            data = pickle.load(handle)
        print(':: <gather> [2/2] Reading from' , metadataf)
        with open(metadataf, 'rb') as handle:
            metadata = pickle.load(handle)
        return data, metadata

    # else
    #  process raw data and return
    return process(tag, window_size)


def process_file(filename, run_tests=True, window_size=None):
    # fetch data from file
    print(':: <proc> [1/3] Fetch data from file')
    data = fetch_data(BASE_PATH + '/' + filename)

    print(':: <proc> [2/3] Preprocess data')
    data = preprocess(data)

    if window_size:
        print(':: <proc> [3/3] Get windows')
        windows = build_windows(data, window_size=5)

        # integrate windows to dataset
        data['windows'] = windows


    if run_tests:
        print(':: <test> [1/2] Test preprocessed data')
        test_preprocessed_dataset(data)

        if window_size:
            print(':: <test> [2/2] Test windows')
            test_windows(windows, window_size=5)

    return data


if __name__ == '__main__':
    data, metadata = gather(TYPE_VERB)
    data, metadata = gather(TYPE_PREP, window_size=None)
    data, metadata = gather(TYPE_NE)
    data, metadata = gather(TYPE_NOUN)
