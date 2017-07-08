import numpy as np

from sklearn import cross_validation, metrics

from itertools import chain
from six.moves import range, reduce

from tasks.babi.proc import load_task, vectorize_data


def fetch(task_id, batch_size, datadir):

    # data directory
    datadir = datadir + '/en/'
    # task data
    train, test = load_task(datadir, task_id)
    data = train + test

    # metadata
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    # sizes
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(50, max_story_size)
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position

    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=None)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    # params
    n_train = trainS.shape[0]
    n_test = testS.shape[0]
    n_val = valS.shape[0]

    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]

    data = {
        'trS' : trainS,
        'trQ' : trainQ,
        'trA' : trainA,
        'teS' : testS,
        'teQ' : testQ,
        'teA' : testA,
        'vaS' : valS,
        'vaQ' : valQ,
        'vaA' : valA,
        'batches' : batches
        }


    metadata = {
            'vocab_size' : vocab_size,
            'vocab' : vocab,
            'word_idx' : word_idx,
            'sentence_size' : sentence_size,
            'memory_size' : memory_size
            }

    return data, metadata
