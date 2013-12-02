from n_grams import build_feature_index_map, features_from_list_of_bars
import numpy as np
import copy
from sessionparse import *
from cvxopt import solvers, matrix, spdiag, lapack, spmatrix
import cPickle

def mat_to_col_major_order(a):
    return a.reshape((a.size, 1), order='F')

def col_major_order_to_mat(a):
    n = int(round(np.sqrt(a.size)))
    return a.reshape((n, n), order='F')

def save_cvx_params(ys):
    n, m = ys.shape

    cmat = np.zeros((n, n))

    for j in xrange(n):
        for i in xrange(j, n):
            val = ys[i, :].dot(ys[j, :])
            cmat[i, j] = val
            cmat[j, i] = val

    import scipy.io
    scipy.io.savemat('cmat.mat',{'cmat': cmat})

def recover_matrix_from_soln(x, index_dict_flip, n):
    xmat = np.zeros((n, n))

    for kk, val in enumerate(np.array(x)):
        i, j = index_dict_flip[kk]
        xmat[i, j] = val
        xmat[j, i] = val

    return xmat

def ys_from_pairs(pairs):
    n = len(pairs[0][0])
    x_list = [np.reshape(p[0] - p[1], newshape = (n)) for p in pairs]
    x_list = [x for x in x_list if np.linalg.norm(x) > 10 ** (-6)]
    
    x_np = np.zeros((n, len(x_list)))
    for i, x in enumerate(x_list):
        x_np[:, i] = x

    return x_np

def build_a_b_pairs_vector(n = 2, num_blocks = 6):
    ''' 
    Build feature vectors for each half of each tune    
    We hard code 0 as A section, 1 as B section 
    '''
    assert(num_blocks >= 1 and num_blocks <= 6)

    pairs = list()

    d = build_feature_index_map(n)

    for i in xrange(num_blocks):
        tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'.format(i), 'rb'))

        for tune in tunes:
            try:
                a, b = ab_split(tune)
            except:
                continue

            a_sec = features_from_list_of_bars(a, d, n)
            b_sec = features_from_list_of_bars(b, d, n)

            pairs.append((a_sec, b_sec))

    return pairs
