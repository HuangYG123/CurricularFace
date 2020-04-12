"""Common metrics used for evaluation
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from warnings import warn

# Find thresholds given FARs
# but the real FARs using these thresholds could be different
# the exact FARs need to recomputed using calcROC
def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds



def ROC(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    ''' Compute Receiver operating characteristic (ROC) with a score and label vector.
    '''
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)

    assert len(thresholds.shape)==1 
    if np.size(thresholds) > 10000:
        warn('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i,threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = np.mean(accept[label_vec])
        FARs[i] = np.mean(accept[~label_vec])
        if get_false_indices:
            false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
            false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds

def ROC_by_mat(score_mat, label_mat, thresholds=None, FARs=None, get_false_indices=False, triu_k=None):
    ''' Compute ROC using a pairwise score matrix and a corresponding label matrix.
        A wapper of ROC function.
    '''
    assert score_mat.ndim == 2
    assert score_mat.shape == label_mat.shape
    assert label_mat.dtype == np.bool
    
    # Convert into vectors
    m,n  = score_mat.shape
    if triu_k is not None:
        assert m==n, "If using triu for ROC, the score matrix must be a sqaure matrix!"
        triu_indices = np.triu_indices(m, triu_k)
        score_vec = score_mat[triu_indices]
        label_vec = label_mat[triu_indices]
    else:
        score_vec = score_mat.flatten()
        label_vec = label_mat.flatten()

    # Compute ROC
    if get_false_indices:
        TARs, FARs, thresholds, false_accept_indices, false_reject_indices = \
                    ROC(score_vec, label_vec, thresholds, FARs, True)
    else:
        TARs, FARs, thresholds = ROC(score_vec, label_vec, thresholds, FARs, False)

    # Convert false accept/reject indices into [row, col] indices
    if get_false_indices:
        rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        rc = np.stack([rows, cols], axis=2)
        if triu_k is not None:
            rc = rc[triu_indices,:]
        else:
            rc = rc.reshape([-1,2])

        for i in range(len(FARs)):
            false_accept_indices[i] = rc[false_accept_indices[i]]
            false_reject_indices[i] = rc[false_reject_indices[i]]
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds




def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False):
    ''' Closed/Open-set Identification. 
        A general case of Cummulative Match Characteristic (CMC) 
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks, 
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape==label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[match_indices,:]
    label_mat_m = label_mat[match_indices,:]
    score_mat_nm = score_mat[np.logical_not(match_indices),:]
    label_mat_nm = label_mat[np.logical_not(match_indices),:]

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]
        
    # Calculate DIRs for different FARs and ranks
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:,0:rank].any(axis=1)
            DIRs[i,j] = (score_rank & retrieval_rank).astype(np.float32).mean()
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    return DIRs, FARs, thresholds

def accuracy(score_vec, label_vec, thresholds=None):
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype==np.bool
    # find thresholds by TAR
    if thresholds is None:
        score_pos = score_vec[label_vec==True]
        thresholds = np.sort(score_pos)[::1]    

    assert len(thresholds.shape)==1
    if np.size(thresholds) > 10000:
        warn('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))
    
    # Loop Computation
    accuracies = np.zeros(np.size(thresholds))
    for i, threshold in enumerate(thresholds):
        pred_vec = score_vec>=threshold
        accuracies[i] = np.mean(pred_vec==label_vec)

    # Matrix Computation, Each column is a threshold
    # predictions = score_vec[:,None] >= thresholds[None,:]
    # accuracies = np.mean(predictions==label_vec[:,None], axis=0)

    argmax = np.argmax(accuracies)
    accuracy = accuracies[argmax]
    threshold = np.mean(thresholds[accuracies==accuracy])

    return accuracy, threshold
