import sys
sys.path.append('../../')

from tasks.cbt.pipeline import *
from tproc.utils import *

from tqdm import tqdm

FIELDS = [ 'context', 'query', 'candidates', 'answer' ]
#DSETS  = [ 'train', 'test', 'valid' ]
DSETS  = [ 'test', 'valid' ]
PICKLES = [ 'data.NE', 'lookup.NE', 'metadata.NE' ]

PATH = '../../../datasets/CBTest/data/'


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
    orig_sample = sample

    # get candidates
    candidates = sample.splitlines()[-1].split('\t')[-1].split('|')
    answer = sample.splitlines()[-1].split('\t')[1].strip()

    # filter candidates
    candidates = [ w for w in candidates if is_worthy(w) ]

    # update vocab
    for w in candidates:
        if w not in vocab and is_worthy(w):
            vocab.append(w)

    # add unk to candidates to keep shape at 10
    candidates = candidates + ['UNK']*(10 - len(candidates))

    # build word to index
    # w2i = { w:i for i,w in enumerate(vocab) }

    # assign special tokens to candidates
    for w in candidates:
        token_w = 'cand' + str(vocab.index(w))
        candidate_lookup[token_w] = w
        sample = sample.replace(w, token_w)

    #sample = rtext_pipeline(sample)

    # update vocabulary
    #  obtain words from string(sample)
    # split sample into story and query
    story = ' '.join([ line.lstrip('0123456789') for line in sample.splitlines()[:-1] ])
    query = sample.splitlines()[-1].split('\t')[0].lstrip('21')

    # run through raw text pipeline
    story = rtext_pipeline(story)
    query = rtext_pipeline(query)


    for w in (story + ' ' + query).split():
        if w not in vocab:
            vocab.append(w)

    # vectorize sample
    data = {
            'context' : vectorize_tree(story.split(), vocab, is_worthy),
            'query' : vectorize_tree(query.split(), vocab, is_worthy),
            'candidates' : vectorize_tree(candidates, vocab, is_worthy),
            'answer' : vectorize_tree(answer, vocab, is_worthy)
            }


    # replace special tokens in vocab with actual words
    #for token, w in candidate_lookup.items():
    #    vocab[vocab.index(token)] = w

    return data, vocab, candidate_lookup


# check if a word is worthy
def is_worthy(w):
    return len(w) > 0 and len(w) < 30 and 'http' not in w and 'www' not in w


# init metadata
def init_metadata():
    return { 
            'max_candidates' : 10,
            'clen' : 0,
            'qlen' : 0
            }

# update metadata
def update_metadata(metadata, data_item):
    return {
            'clen' : max(metadata['clen'], len(data_item['context'])),
            'qlen' : max(metadata['clen'], len(data_item['query']))
            }


def process_file(filepath, vocab, metadata):
    # fetch samples from file
    samples = fetch_samples(filepath)

    # maintain vocabulary
    candidate_lookup = {}

    data = {}
    for k in FIELDS:
        data[k] = []

    # iterate through samples
    for sample in tqdm(samples):
        data_item, vocab, candidate_lookup = process_sample(sample,
                vocab, candidate_lookup)

        # update metadata
        metadata = update_metadata(metadata, data_item)

        for k in FIELDS:
            data[k].append(data_item[k])

    return data, vocab, metadata


def process():

    # currently we are working only on named entities
    filepath = {
            'train' : '../../../datasets/CBTest/data/cbtest_NE_train.txt',
            'test'  : '../../../datasets/CBTest/data/cbtest_NE_valid_2000ex.txt',
            'valid' : '../../../datasets/CBTest/data/cbtest_NE_test_2500ex.txt'
            }

    # maintain
    #  1. data
    #  2. (update) lookup -> vocabulary
    #  3. (update) metadata
    #    for each tag (train, test, valid)
    data, metadata = {}, init_metadata()
    lookup = ['PAD', 'UNK']
    for tag in DSETS:
        data_, lookup, metadata = process_file(filepath[tag], lookup, metadata)
        data[tag] = data_

    # update lookup size in metadata
    metadata['vocab_size'] = len(lookup)

    # create w2i and i2w
    w2i = { w:i for i,w in enumerate(lookup) }
    i2w = lookup

    lookup = { 'w2i' : w2i, 'i2w' :i2w }

    # save to disk
    serialize(data, PATH + 'data.NE')
    serialize(lookup, PATH + 'lookup.NE')
    serialize(metadata, PATH + 'metadata.NE')

    return data, lookup, metadata

def gather():

    for pickle_file in PICKLES:
        if os.path.isfile(PATH + pickle_file):
            continue
        else:
            return process()

    return [read_pickle(PATH + pfile) for pfile in PICKLES]


if __name__ == '__main__':
    # run process -> get data
    process()
