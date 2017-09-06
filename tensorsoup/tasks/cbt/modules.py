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
