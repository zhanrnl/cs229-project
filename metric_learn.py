from n_grams import build_feature_index_map, features_from_list_of_bars
import numpy as np
import copy
from sessionparse import *
from cvxopt import solvers, matrix, spdiag

def mat_to_col_major_order(a):
    return a.reshape((a.size, 1), order='F')

def col_major_order_to_mat(a):
    n = int(round(np.sqrt(a.size)))
    return a.reshape((n, n), order='F')

def sdp_params(ys):
    n = len(ys[0])
    c = np.zeros((n, n))

    for yk in ys:
        for i in xrange(n):
            for j in xrange(n):
                c[i, j] += yk[i] * yk[j]

    c = matrix(mat_to_col_major_order(c))

    G1 = spdiag(matrix(-1 * np.ones(n ** 2)))
    h1 = spdiag(matrix(np.zeros(n)))

    Gs = [G1]
    hs = [h1]

    return c, Gs, hs

def ys_from_pairs(pairs):
    n = len(pairs[0][0])
    x_list = [np.reshape(p[0] - p[1], newshape = (n, 1)) for p in pairs]
    x_list = [x for x in x_list if np.linalg.norm(x) > 10 ** (-6)]
    
    return x_list

'''
def mahalanobis_learn(pairs):
    n = len(pairs[0][0])
    A = np.eye(n)

    x_list = [np.reshape(p[0] - p[1], newshape = (n, 1)) for p in pairs]
    x_list = [x for x in x_list if np.linalg.norm(x) > 10 ** (-6)]

    x_inner_prod_sum = 0
    for x in x_list:
        for y in x_list:
            x_inner_prod_sum += x.T.dot(y)

    x_sum_mat = np.zeros(n, n)
    for x in x_list:
        x_sum_mat += x.dot(x.T)

    x_sum_mat_norm = 0


    for i in xrange(1):
        import time
        s = time.time()
        update_A = copy.deepcopy(A)
        for x in x_list:
            update_A += (x.T.dot(A.dot(x))) ** (-0.5) * x.dot(x.T)
        A += update_A
        print 'Took {0} s'.format(time.time() - s)
    return A
'''

def build_a_b_pairs_vector(n = 2, num_blocks = 6):
    ''' We hard code 0 as A section, 1 as B section '''
    import cPickle

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
