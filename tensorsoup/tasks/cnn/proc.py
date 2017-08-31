import sys
import logging 
sys.path.append('../../')

ROOT= '../../../datasets/cnn/'

#corresponding directories
TRAIN, TEST, VALID = 'training', 'test', 'validation'
DSETS = TRAIN, TEST, VALID

import logging
log = logging.getLogger('tasks.cnn.proc')
log.setLevel(logging.DEBUG)
from pprint import pprint, pformat

import os
from pprint import pprint
from tqdm import tqdm
from tproc.utils import preprocess_text
from tproc.dictionary import Dictionary, buildDictionary

def process_data(rootdir, dset=TEST, nsamples=0):
    '''
    Contexts, Questions, Candidates, Answers, Origwords = fetch_data(ROOT)
    where ROOT dir contains questions/training/*.question
    '''
    print(locals())
    samples = []
    dirname = rootdir + '/questions/' + dset
    Contexts, Questions, Candidates, Answers, Origwords = [], [], [], [], []
    for samplecount, filename in tqdm(enumerate(os.listdir(dirname))):
        if nsamples and samplecount > nsamples :
            break
        
        with open(dirname+'/'+filename) as sample:
            lines = sample.read().splitlines()
            url, _, context, _, question, _, answer, _, *__candidates = lines
            candidates = []
            origwords = []
            for c in __candidates:
                candidate, origword = c.split(':', 1)
                candidates.append(candidate)
                origwords.append(origword)
                
        context, question, answer = [i.split() for i in 
                                     [context, question, answer]]
        
        Contexts  .append( context  )        
        Questions .append( question )
        Answers   .append( answer   )
        
        Candidates.append( candidates         )
        Origwords .append( origwords          )
        
    print('------url------------\n', url)
    print('-------question--------------\n', question)
    print('---------candidates-----------------\n', candidates)
    print('--------origwords-----------------\n', origwords)
    print('---------answer------------------\n', answer)
    print('---------context-----------------\n', context)
    
    return Contexts, Questions, Candidates, Answers, Origwords


import pickle
def pickleSet(dirname, contexts, questions, candidates, answers, origwords):
    '''
    pickleSet(ROOT+'/processed_questions/test', 
          Contexts, Questions, Candidates, Answers, Origwords)
    '''
    names = 'contexts', 'questions', 'candidates', 'answers', 'origwords'
    data = contexts, questions, candidates, answers, origwords
    
    for name, datum in zip(names, data):
        with open(dirname+'/'+name, 'wb') as f:
            pickle.dump(datum, f,  pickle.HIGHEST_PROTOCOL)
            
def loadSet(dirname, nsamples):
    '''
    loadSet(ROOT+'/processed_questions/'+tag)
    '''
    names = 'contexts', 'questions', 'candidates', 'answers', 'origwords'
    data = []
    for name in names:
        with open(dirname+'/'+name, 'rb') as f:
            data.append(pickle.load(f))
            
    return [d[:nsamples] for d in data]


import os.path
def load_data(root, dset, dformat=None, nsamples=10000):

    processed_dir = root+'/processed_questions'
    dset_dir = processed_dir + '/' + dset
    print('info>> load_data: {}'.format(dset_dir))
    if not os.path.exists(dset_dir):
        if not os.path.exists(processed_dir): os.mkdir(processed_dir, 0o755)
        contexts, questions, candidates, answers, origwords = process_data(root, dset, nsamples)
        os.mkdir(dset_dir, 0o755)
        pickleSet(processed_dir+'/'+dset, 
                  contexts, questions, candidates, answers, origwords)
        
    else:

        contexts, questions, candidates, answers, origwords = loadSet(dset_dir, nsamples)

    data = {}
    data['contexts']    = contexts
    data['questions']  = questions
    data['candidates'] = candidates
    data['answers']    = answers
    data['origwords']  = origwords

    if dformat:
        data = { key:data[key] for key in dformat }
    
    return data

