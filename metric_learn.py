from n_grams import build_feature_index_map, features_from_list_of_bars
import numpy as np
import copy
from sessionparse import *
from cvxopt import solvers, matrix, spdiag, lapack, spmatrix
import cPickle
import scipy.io
import subprocess
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree, DistanceMetric

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
        print "Loading pickle file {}... ".format(i)
        tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'.format(i), 'rb'))
        print "  loaded."

        for tune in tunes:
            try:
                a, b = ab_split(tune)
            except:
                continue

            a_sec = features_from_list_of_bars(a, d, n)
            b_sec = features_from_list_of_bars(b, d, n)

            pairs.append((a_sec, b_sec))

    return pairs

def metric_learn(pairs_test, pairs_train, n_components = 35):
    pca = PCA(n_components = n_components)
    pca.fit(pairs_train_flat)

    pairs_flat = [item for subtuple in pairs for item in subtuple]
    pairs_pca_flat = pca.transform(pairs_flat)
    
    pairs_pca = list()
    for i in xrange(0, len(pairs_pca_flat), 2):
        a = i 
        b = i + 1
        pairs_pca.append((pairs_pca_flat[a], pairs_pca_flat[b]))
    
    pairs_pca_train = pairs_pca[:n_train]
    pairs_pca_test = pairs_pca[n_train:]

    ys = ys_from_pairs(pairs_pca_train)

    save_cvx_params(ys)
    run_cvx()
    print 'finished running cvx'

    M = load_cvx_result()

    return pairs_pca_train, pairs_pca_test, M

def neighbors(n_components = 35, fraction = 0.10):
    pairs_pca_train, pairs_pca_test, M = metric_learn(n_components)

    dist = DistanceMetric.get_metric('mahalanobis', VI = M)

    a_sections = [x[0] for x in pairs_pca_test]
    b_sections = [x[1] for x in pairs_pca_test]

    a_sections_train = [x[0] for x in pairs_pca_train]
    b_sections_train = [x[1] for x in pairs_pca_train]
    
    bt = BallTree(a_sections, metric = dist)
    bt_train = BallTree(a_sections_train, metric = dist)
    bt_euc = BallTree(a_sections)
    bt_euc_train = BallTree(a_sections_train)

    top_fraction = int(len(b_sections) * fraction)
    top_fraction_train = int(len(b_sections_train) * fraction)

    res = bt.query(b_sections, top_fraction)
    res_train = bt_train.query(b_sections_train, top_fraction_train)
    res_euc = bt_euc.query(b_sections, top_fraction)
    res_euc_train = bt_euc_train.query(b_sections_train, top_fraction_train)

    c = 0
    c_euc = 0
    for i in xrange(len(b_sections)):
        if i in res[1][i]:
            c += 1
        if i in res_euc[1][i]:
            c_euc += 1

    c_train = 0
    c_euc_train = 0
    for i in xrange(len(b_sections_train)):
        if i in res_train[1][i]:
            c_train += 1
        if i in res_euc_train[1][i]:
            c_euc_train += 1

    return ((c, c_euc, len(b_sections)), (c_train, c_euc_train, len(b_sections_train)))
