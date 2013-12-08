from n_grams import build_feature_index_map, features_from_list_of_bars
import numpy as np
import copy
from sessionparse import *
from cvxopt import solvers, matrix, spdiag, lapack, spmatrix
import cPickle
import scipy.io
import subprocess
from sklearn.decomposition import PCA

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

    scipy.io.savemat('cmat.mat',{'cmat': cmat})

def run_cvx():
    subprocess.call(['matlab', '-nodisplay', '-nosplash', '-nodesktop', '-r', 'try, sdp_solve, catch, exit, end, exit'])

def load_cvx_result():
    result = scipy.io.loadmat('result.mat')
    M = result['M']
    return M

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

def metric_learn():
    pairs = build_a_b_pairs_vector(2, 6)
    n_train = int(round(0.7 * len(pairs)))

    pairs_train = pairs[:n_train]
    pairs_train_flat = [item for subtuple in pairs_train for item in subtuple]
    
    pca = PCA(n_components = 35)
    pca.fit(pairs_train_flat)

    pairs_flat = [item for subtuple in pairs_train for item in subtuple]
    pairs_pca_flat = pca.transform(pairs_flat)
    
    pairs_pca = list()
    for i in xrange(0, len(pairs_pca_flat), 2):
        a = i 
        b = i + 1
        pairs_pca.append((pairs_pca_flat[a], pairs_pca_flat[b]))
    
    pairs_pca_train = pairs_pca[:n_train]

    ys = ys_from_pairs(pairs_pca_train)

    save_cvx_params(ys)
    run_cvx()
    print 'finished running cvx'

    M = load_cvx_result()
