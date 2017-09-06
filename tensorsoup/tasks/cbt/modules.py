import numpy as np

# context -> paddeded numpy array [N, clen]
# candidates -> [N, 10]
def candidate_mask(contexts, candidates):
    
    mask = []
    for context, candidates_i in zip(contexts, candidates):
        mask_i = []
        for candidate in candidates_i:
            mask_i.append(context == candidate)
        mask.append(mask_i)

    return np.array(mask, dtype=np.int32)

def reindex_answer(answer, candidates):
    return [ c.index(a) for a,c in zip(answer, candidates) ]

# sort based on len of context
#  data : dictionary of padded numpy arrays
def sort_data(data):
    context = data['context']
    n = len(context)

    # sorted based on num zeros (padding symbols)
    idx = [ w for w,x in sorted(enumerate(context), 
        key=lambda x : np.count_nonzero(x[1])) ]

    return { k:v[idx] for k,v in data.items() }
