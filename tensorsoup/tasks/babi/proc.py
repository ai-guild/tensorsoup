import os
import sys

sys.path.append('../../')

from tproc.utils import *


DATA_DIR_1K = '../../../datasets/babi/en/'
DATA_DIR_10K = '../../../datasets/babi/en-10k/'


def read_from_file(filename):
    with open(filename) as f:
        return [ line for line in f.read().split('\n') if line ]

def fetch_data(filename):
    contexts, answers, supporting_facts = [], [], []
    questions = []
    context = []

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
    def printv(*s):
        if verbose == True:
            print(*s)
            
    def process_sample(sample):

        def sanitize(s):
            s = ' '.join(s.split(' ')[1:])
            return str(s).strip('.? \t\n')
            
        def process_support_old(support):
            s =  support.split(' ')
            s = [sample_dict[i] for i in s]
            
            return [local_context.index(i) for i in s]

        def process_support(support, debug=False):
            s = support.split(' ')
            s = [int(i) for i in s]
            _support = []
            for j in s:
                for i, line in enumerate(local_context):
                    index = int(line.split(' ')[0])
                    #printv(i, j, index, line)
                    if j == index:
                        if debug:
                            _support.append((i, j))
                        else:
                            _support.append(i)
                        break

                        
            #printv(_support)
            #printv('=================')
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

                support = process_support(support)

                questions.append(question)
                answers.append(answer)
                supports.append(support)
                
            else:
                local_context.append(line)

        contexts = [[sanitize(s) for s in context] for context in contexts]
        questions = [sanitize(question) for question in questions]
        
        return contexts, questions, answers, supports

    contexts, questions, answers, supports = [], [], [], []
    temp  = 0
    from pprint import pprint
    for sample in samples:
        c, q, a, s = process_sample(sample)
        print('len ==>', len(c), len(q), len(a), len(s))
        pprint([c, q, a, s])
        print('########')
        contexts.extend(c)
        questions.extend(q)
        answers.extend(a)
        supports.extend(s)

    return contexts, questions, answers, supports


def process():

    # get list of files
    files = list_of_files(DATA_DIR_1K)

    #raw_data = [ fetch_data(f) for f in files ]

    return fetch_data1(files[0])
