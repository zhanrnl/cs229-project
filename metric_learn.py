from n_grams import build_feature_index_map, features_from_list_of_bars
import numpy as np
import copy
from sessionparse import *

def mahalanobis_learn(pairs):
    n = len(pairs[0][0])
    A = np.eye(n)

    x_list = [np.reshape(p[0] - p[1], newshape = (n, 1)) for p in pairs]
    x_list = [x for x in x_list if np.linalg.norm(x) > 10 ** (-6)]

    for i in xrange(1):
        import time
        s = time.time()
        update_A = copy.deepcopy(A)
        for x in x_list:
            update_A += (x.T.dot(A.dot(x))) ** (-0.5) * x.dot(x.T)
        A += update_A
        print 'Took {0} s'.format(time.time() - s)
    return A

def build_a_b_pairs_vector(n = 2):
    ''' We hard code 0 as A section, 1 as B section '''
    import cPickle

    pairs = list()

    d = build_feature_index_map(n)

    for i in xrange(6):
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
