import sys
sys.path.append('../../')

from tasks.cbt.pipeline import *

# separate items from text
def _read_sample(sample):
    lines = sample.splitlines()
    context = ''
    for line in lines[:-1]:
        context += line.lstrip('0123456789')
    query = lines[-1].split('\t')[0].lstrip('21')
    candidates = lines[-1].split('\t')[-1]
    answer = lines[-1].split('\t')[1]
    return context, query, candidates, answer


# read from file
#  returns structured text
def fetch_samples(path):
    # open file
    with open(path) as f:
        #raw_data = rtext_pipeline(f.read())
        raw_data = f.read()
        samples = raw_data.split('\n\n')[:-1]

    return samples

# text sample to data item
def process_sample(sample, vocab, candidate_lookup):
    # sample -> raw text

    # get candidates
    candidates = sample.splitlines()[-1].split('\t')[-1].split('|')
    answer = sample.splitlines()[-1].split('\t')[1].strip()

    # update vocab
    for w in candidates:
        if w not in vocab:
            vocab.append(w)

    # build word to index
    # w2i = { w:i for i,w in enumerate(vocab) }

    # assign special tokens to candidates
    for w in candidates:
        token_w = 'cand' + str(vocab.index(w))
        candidate_lookup[token_w] = w
        sample = sample.replace(w, token_w)

    # run through raw text pipeline
    sample = rtext_pipeline(sample)

    # update vocabulary
    #  obtain words from string(sample)
    # split sample into story and query
    story = ' '.join([ line.lstrip('0123456789') for line in sample.splitlines()[:-1] ])
    query = sample.splitlines()[-1].split('\t')[0].lstrip('21')
    for w in (story + query).split(' '):
        if w not in vocab:
            vocab.append(w)

    # vectorize sample


    # replace special tokens in vocab with actual words
    #for token, w in candidate_lookup.items():
    #    vocab[vocab.index(token)] = w

    return (story, query, candidates, answer), vocab, candidate_lookup



if __name__ == '__main__':
    filepath = '../../../datasets/CBTest/data/cbtest_NE_test_2500ex.txt'
    samples = fetch_samples(filepath)
    #print(samples[119], samples[1300])

    vocab = []
    candidate_lookup = {}
    data, vocab, candidate_lookup = process_sample(samples[113], 
            vocab, candidate_lookup)

    print(data)
    print('______________________________')
    print(vocab)
    print(candidate_lookup)
