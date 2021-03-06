import os
import sys
import pickle

sys.path.append('../../')

from tproc.utils import *


DATA_DIR_1K = '../../../datasets/babi/en/'
DATA_DIR_10K = '../../../datasets/babi/en-10k/'

MAX_MEMORY_SIZE = 50

TRAIN = 'train'
TEST = 'test'
TAGS = [ TRAIN, TEST ]

PAD = '<pad>' 
UNK = '<unk>'
special_tokens = [PAD]

KEYS = [ 'contexts', 'questions', 'answers', 'supports', 'vocab' ]


def read_from_file(filename):
    with open(filename) as f:
        return [ line for line in f.read().split('\n') if line ]

def process_file(filename):
    contexts, answers, supporting_facts = [], [], []
    questions = []
    context = []
    words = [] # list of words for vocabulary

    offset = 0

    # read from file
    lines = read_from_file(filename)
    sample = []
    samples = []
    for i, line in enumerate(lines):
        index = int(line.split(' ')[0])
        if index == 1:
            samples.append(sample)
            sample = [line]
        else:
            sample.append(line)

    samples = samples[1:]
    verbose = True
            
    def process_sample(sample):

        def sanitize(s):
            s = ' '.join(s.split(' ')[1:])
            return str(s).strip('.? \t\n')
            
        def process_support(support, debug=False):
            s = support.split(' ')
            s = [int(i) for i in s]
            _support = []
            for j in s:
                for i, line in enumerate(local_context):
                    index = int(line.split(' ')[0])
                    if j == index:
                        if debug:
                            _support.append((i, j))
                        else:
                            _support.append(i)
                        break

            return _support
        
        contexts, questions, answers, supports = [], [], [], []

        local_context = []
        sample_dict  = {}
        for line in sample:
            index = line.split(' ')[0]
            sample_dict[index] = line

            if '?' in line:
                contexts.append(local_context[:])
                question, answer, support = line.split('\t')
                question = question[:question.index('?')]

                # add words to list
                words.extend(sanitize(question).split(' '))

                support = process_support(support)

                questions.append(question)
                answers.append(answer)
                supports.append(support)
                
            else:
                # add words to list
                words.extend(sanitize(line).split(' '))
                # add line to current context
                local_context.append(line)

        contexts = [[sanitize(s) for s in context] for context in contexts]
        questions = [sanitize(question) for question in questions]
        
        return contexts, questions, answers, supports

    contexts, questions, answers, supports = [], [], [], []
    for sample in samples:
        c, q, a, s = process_sample(sample)
        contexts.extend(c)
        questions.extend(q)
        answers.extend(a)
        supports.extend(s)

    # reverse contexts
    contexts = [ c[::-1] for c in contexts ]

    # build dictionary
    struc_text= {
            'contexts' : contexts,
            'questions' : questions,
            'answers' : answers,
            'supports' : supports,
            'vocab' : list(set(words))
            }


    return struc_text

def gather_metadata(train, test):
    data = {}

    # combine test and train
    for k in train.keys():
        data[k] = train[k] + test[k]

    # combined vocab
    vocab = special_tokens + sorted(list(set(data['vocab'])))

    # candidates vocab
    cvocab = sorted(list(set(data['answers'])))

    return {
            'slen' : max([len(s.split(' ')) 
                for c in data['contexts'] for s in c ]),
            'qlen' : max([len(q.split(' ')) for q in data['questions']]),
            'w2i' : { w:i for i,w in enumerate(vocab) },
            'i2w' : vocab,
            'vocab_size' : len(vocab),
            'clen' : min(max([len(c) for c in data['contexts']]), MAX_MEMORY_SIZE),
            'special_tokens' : special_tokens,
            'candidates' : {
                'vocab_size' : len(cvocab),
                'w2i' : { w:i for i,w in enumerate(cvocab) },
                'i2w' : cvocab
                }
            }

def index(data, metadata):
    # global word to index dictionary
    gw2i = metadata['w2i']
    # w2i for candidates
    cw2i = metadata['candidates']['w2i']

    indexed_data = {}
    for k in ['contexts', 'questions']:
        indexed_data[k] = index_seq(data[k], gw2i)

    # index answers with candidates vocabulary
    indexed_data['answers'] = index_seq(data['answers'], cw2i)

    # add supporting facts
    indexed_data['supports'] = data['supports']

    return indexed_data


def process(path, dtype, serialize=True):

    # get list of files
    #  sort files based on name (task id)
    #   and type of data (train/test)
    files = [ f for f in list_of_files(path) if '.txt' in f ]
    files = sorted(files, key = lambda x : 
            (int(x.split('/')[-1].split('_')[0][2:]), x.split('_')[-1] ))

    # get test and train files given task id
    get_files_by_task = lambda tid : files[(tid-1)*2 : (tid-1)*2 + 2]

    # init data dict 
    #  that holds separate and joined tasks
    data = {
            'train' : { k:[] for k in KEYS },
            'test'  : { k:[] for k in KEYS }
            }

    # process 20 tasks
    struc_texts_test = [] # plural -> list
    for i in range(1, 21):
        # process train and test files
        testfile, trainfile = get_files_by_task(i)
        struc_text_test = process_file(testfile)
        struc_text_train = process_file(trainfile)

        # get metadata
        metadata = gather_metadata(struc_text_train, 
                struc_text_test)

        # append to list (to be used later)
        struc_texts_test.append(struc_text_test)

        # index structured text
        data[i] = {
                'train' : index(struc_text_train, metadata),
                'test'  : index(struc_text_test , metadata),
                'metadata' : metadata
                }

        # combine text data
        for k in KEYS:
            data['train'][k].extend(struc_text_train[k])
            data['test'][k].extend(struc_text_test[k])

    # gather metadata for joined tasks
    metadata = gather_metadata(data['train'], data['test'])

    # index combined data
    for tag in TAGS:
        data[tag] = index(data[tag], metadata)

    # add test set separate tasks to data['test']
    #  reindex individual tasks data
    for j in range(1, 21):
        # reindex with global metadata
        data['test'][j] = index(struc_texts_test[j-1], metadata)

    if serialize:
        print(':: [1/1] Serialize data and metadata')
        with open('{}/data.{}'.format(path, dtype), 'wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
        with open('{}/metadata.{}'.format(path, dtype), 'wb') as handle:
            pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)

    return data, metadata


def gather(dtype, task=0):

    path = {
            '1k' : DATA_DIR_1K,
            '10k' : DATA_DIR_10K
            }

    # build file names
    dataf = '{}/data.{}'.format(path[dtype], dtype)
    metadataf = '{}/metadata.{}'.format(path[dtype], dtype)

    # if processed files exist
    #  read pickle and return
    if os.path.isfile(dataf) and os.path.isfile(metadataf):
        print(':: <gather> [1/2] Reading from' , dataf)
        with open(dataf, 'rb') as handle:
            data = pickle.load(handle)
        print(':: <gather> [2/2] Reading from' , metadataf)
        with open(metadataf, 'rb') as handle:
            metadata = pickle.load(handle)
    else:
        #  process raw data and return
        data, metadata = process(path[dtype], dtype)

    # check task > 0
    #  return separate tasks
    if task > 0:
        metadata = data[task]['metadata']
        data = { 'train' : pad(data[task]['train'], metadata),
                'test' : pad(data[task]['test'], metadata)
                }

    else:
        # task = 0
        train = pad(data['train'], metadata)
        test = { k:pad(data['test'][k], metadata) for k in range(1,21) }
        test.update(pad(data['test'], metadata))
        data = { 'train' : train, 'test' : test}


    return data, metadata
        

def pad(data, metadata):
    clen, slen, qlen = [ metadata[k] for k in ['clen', 'slen', 'qlen' ] ]
    padded_data = {}

    memory_size = min(clen, MAX_MEMORY_SIZE)

    padded_data = { 
        'contexts' : pad_sequences(data['contexts'],
            maxlens = [memory_size, slen], metadata=metadata),
        'questions' : pad_sequences(data['questions'],
            maxlens = [0, qlen], metadata=metadata),
        'answers' : np.array(data['answers']),
        'supports' : np.array(data['supports'])
        }

    # TODO : separate shuffle from pad
    return shuffle(padded_data)



if __name__ == '__main__':
    data, metadata = gather('1k', 1)
    data, metadata = gather('10k', 20)
    print(data.keys())
    print(metadata.keys())
