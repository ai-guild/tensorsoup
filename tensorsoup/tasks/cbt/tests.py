import sys


def test_windows(windows, window_size):
    # shape check
    wlens = [len(wj) for wi in windows for wj in wi]
    assert max(wlens) == min(wlens) == window_size


def test_vocabulary(vocab, dataset):
    answers, candidates = dataset['answers'], dataset['candidates']

    # flatten candidates and get unique words
    candidates = list(set([ci for c in candidates for ci in c]))
    # repeat for answers
    answers = list(set(answers))

    # check if all answers exist in candidates
    outliers = [ w for w in answers if w not in candidates ]
    assert len(outliers) == 0

    # check if all candidates exist in vocabulary
    outliers = [ w for w in candidates if w not in vocab ]
    assert len(outliers) == 0


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


