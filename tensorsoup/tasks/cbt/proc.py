BASE_PATH = '../../../datasets/CBTest/data'
TRAIN_FILE = 'cbtest_NE_train.txt'
VALID_FILE = 'cbtest_NE_valid_2000ex.txt'


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
        print(':: > [{}/3] preprocessing text'.format(i))
        processed.append([ preprocess_text(item) 
            for item in data ])

    # preprocess candidates
    print(':: > [0/1] preprocessing candidates')
    candidates = [ preprocess_candidates(c) for c in candidates ]


    dataset = { 'stories' : processed[0],
                'queries' : processed[1],
                'answers' : processed[2]
               }

    dataset['candidates'] = candidates

    print(':: > preprocessing complete; returning dataset')
    return dataset


def build_vocabulary(dataset):
    # get stories and queries
    stories, queries = dataset['stories'], dataset['queries']

    # combine stories and queries
    #  into a text blob
    text = ' '.join(stories + queries)
    
    # get all words
    words = text.split(' ')

    # get unique words -> vocab
    return sorted(list(set(words)))


def test_vocabulary(vocab, dataset):
    answers, candidates = dataset['answers'], dataset['candidates']

    # flatten candidates and get unique words
    candidates = list(set([ci for c in candidates for ci in c]))
    # repeat for answers
    answers = list(set(answers))

    # check if all answers exist in candidates
    assert len([ w for w in answers if w not in candidates ]) == 0

    # check if all candidates exist in vocabulary
    assert len([ w for w in candidates if w not in vocab ]) == 0

    return vocab


def test_preprocessed_dataset(dataset):

    # len check
    dlens = [ len(di) for di in list(dataset.values()) ]
    assert max(dlens) == min(dlens)

    # shape check
    #  answers
    alens = [ len(ai.split(' ')) for ai in dataset['answers'] ]
    assert max(alens) == min(alens) == 1

    # shape check
    #  candidates
    clens = [ len(ci) for ci in dataset['candidates'] ]
    assert max(clens) == min(clens) == 10


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


def test_windows(windows, window_size):
    # shape check
    wlens = [len(wj) for wi in windows for wj in wi]
    assert max(wlens) == min(wlens) == window_size


def process():
    print(':: [0/7] Fetch data from file')
    # fetch data from file
    dataset = fetch_data(BASE_PATH + '/' + TRAIN_FILE)

    print(':: [1/7] Preprocess data')
    dataset = preprocess(dataset)

    print(':: [2/7] test preprocessed data')
    test_preprocessed_dataset(dataset)

    print(':: [3/7] Build vocabulary')
    vocab = build_vocabulary(dataset)

    print(':: [4/7] test vocabulary')
    test_vocabulary(vocab, dataset)

    print(':: [5/7] Get windows')
    windows = build_windows(dataset, window_size=5)

    print(':: [6/7] Test windows')
    test_windows(windows, window_size=5)

    # integrate windows to dataset
    dataset['windows'] = windows

    # create metadata
    metadata = { 'vocab' : vocab }

    print(':: [7/7] Return data and metadata')
    return dataset, metadata


if __name__ == '__main__':
    dataset, metadata = process()
