import csv
import lz78
import numpy as np
from scipy.spatial.distance import *
from scipy.cluster.hierarchy import *
import time
import matplotlib.pyplot as plt
from itertools import izip

def parse_all_csv_dumb():
    csv.register_dialect('thesession', quoting=csv.QUOTE_MINIMAL,
    doublequote=False, escapechar='\\')
    with open('thesession-data/tunes.csv', 'r') as csvfile:
        return list(csv.DictReader(csvfile, dialect='thesession'))

def build_dataset():
    l = parse_all_csv_dumb()
    
    names = list()
    data = list()
    types = list()

    c = 0
    for tune in l:
        segments = [s.rstrip() for s in tune['abc'].split('\n')]
        counter = range(len(segments))
        
        for ind, segment in zip(counter, segments):
            if len(segment) > 0:
                names.append('{0}_{1}'.format(c, ind))
                data.append(segment)
                types.append(tune['type'])

        c += 1
    
    return names, data, types

def build_dist_matrix(names, data, n):
    d = np.zeros(shape=(n,n))
    for i in xrange(n):
        for j in xrange(n):
            if j != i:
                d[i,j] = lz78.ncd(data[i], data[j])

    y = squareform(d)
    return y
   
def make_picture(names, y, n):
    Z = linkage(y)

    plt.figure()
    dendrogram(Z, labels=names[:n])
    plt.ylim(0.5, 1)
    plt.show()

def main(n):
    names, data, types = build_dataset()
    y = build_dist_matrix(names, data, n)
    make_picture(names, y, n)

def nearest_neighbor(x, data, types):
    min_dist = 1
    best_i = 0

    for i, segment in enumerate(data):
        dist = lz78.ncd(x, segment)
        if dist < min_dist:
            min_dist = dist
            best_i = i

    return types[best_i]

def test(n=None):
    names, data, types = build_dataset()

    if n != None:
        names = names[:n]
        data = data[:n]
        types = types[:n]

    count = len(data)
    num_training = int(0.7 * count)
    '''
    import random
    train_indicies = random.sample(xrange(count), num_training)
    test_indicies = [i for i in xrange(count) if i not in set(train_indicies)]

    train_data = [data[i] for i in train_indicies]
    train_types = [types[i] for i in train_indicies]

    test_data = [data[i] for i in test_indicies]
    test_types = [types[i] for i in test_indicies]
    '''

    train_data = data[:num_training]
    train_types = types[:num_training]

    test_data = data[num_training:count]
    test_types = types[num_training:count]

    predicted_types = list()

    count = 0

    start = time.time()
    for d, t in izip(test_data, test_types):
        predicted_types.append(nearest_neighbor(d, train_data, train_types))
        
        count += 1

        if count % 10 == 0:
            print '{0} complete, averaging {1} per second. Estimated {2} seconds remaining'.format(count, count / (time.time() - start), (len(test_data) - count) * (time.time() - start) / count)

    return test_data, test_types, predicted_types

def get_types():
    names, data, types = build_dataset()
    return list(set(types))

def breakdown_test_results(test_data, test_types, predicted_types, types):
    # d gives index in types list for each type
    d = {}
    for i, x in enumerate(types):
        d[x] = i
    
    n = len(types)
    arr = np.zeros(shape=(n,n))

    for i, nn_type in enumerate(predicted_types):
        arr[d[test_types[i]],d[nn_type]] += 1

    print types
    print arr
    print sum(arr)
