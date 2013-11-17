from itertools import islice
import numpy as np
from sessionparse import *

feature_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

def build_n_grams_helper(n_gram_list_list = None):
    ''' 
    Given a list where the nth element is the list of n-grams, returns
    a copy of this list with an (n+1)th element that is the list of (n+1)-grams
    '''
    one_grams = [(x,) for x in feature_list]

    if n_gram_list_list == None:
        return [one_grams]

    longest_grams = n_gram_list_list[-1]
    new_n_grams = list()

    for one_gram in one_grams:
        for big_gram in longest_grams:
            new_n_grams.append(one_gram + big_gram)

    return n_gram_list_list + [new_n_grams]

def build_n_grams(n):
    ''' 
    Creates a list of all m-grams for m <= the argument n
    '''
    curr_list = build_n_grams_helper()

    for i in xrange(n - 1):
        curr_list = build_n_grams_helper(curr_list)

    return [item for sublist in curr_list for item in sublist]

def build_feature_index_map(n = 3):
    ''' 
    Generates a dictionary with keys that are n-grams of notes (1..7),
    and where each value is the corresponding index in our feature vector
    of the count for that n-gram 
    '''

    d = dict()
    for i, n_gram in enumerate(build_n_grams(n)):
        d[n_gram] = i

    return d

def window(seq, n):
    '''
    Returns a sliding window (of width n) over data from hter iterable 
    Shamelessly stolen from StackOverflow
    '''
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def gram_str_from_tuple(gram_tuple):
    '''
    Given a tuple of pitches, returns string of 1..14 to be used as a key
    in the index dictionary
    '''

    return tuple([feature_map[x] for x in gram_tuple])

# LENNART .pitch SOMETIMES RETURNS NOTES INSTEAD OF STRINGS WHY?!?!?!
# (hence the str(x.pitch))
def features_from_list_of_bars(list_of_bars, n_gram_dict, n = 3):
    ''' 
    Given a list of bars (i.e. the 'parsed' element of a Tune) return the
    associated feature vector for all k-grams, k <= n
    '''
    pitch_str = ''.join([str(x.pitch) for b in list_of_bars for x in b])

    # update this later when we add accidentals
    pitch_str = [x for x in pitch_str if x in 'abcdefgABCDEFG']

    features = np.zeros(len(n_gram_dict))

    for k in xrange(1, n + 1):
        for gram in window(pitch_str, k):
            features[n_gram_dict[gram]] += 1

    return features

def build_a_b_features_labels(n = 3):
    ''' We hard code 0 as A section, 1 as B section '''
    import cPickle

    f_vecs = list()
    types = list()

    d = build_feature_index_map(n)

    for i in xrange(6):
        tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'.format(i), 'rb'))

        for tune in tunes:
            try:
                a, b = ab_split(tune)
            except:
                continue

            f_vecs.append(features_from_list_of_bars(a, d, n))
            types.append(0)
            
            f_vecs.append(features_from_list_of_bars(b, d, n))
            types.append(1)

    return f_vecs, types
                

def train_test_split(f_vecs, types, fraction = 0.7):
    assert(fraction > 0 and fraction < 1)

    n_train = int(round(fraction * len(f_vecs)))

    f_train = f_vecs[:n_train]
    t_train = types[:n_train]

    f_test = f_vecs[n_train:]
    t_test = types[n_train:]

    t = (f_train, t_train, f_test, t_test)
    
    return tuple([np.array(x) for x in t])


def a_b_classify(n = 3):
    from sklearn import svm
    f_vecs, types = build_a_b_features_labels(n)

    f_train, t_train, f_test, t_test = train_test_split(f_vecs, types, fraction = 0.9)

    clf = svm.SVC()
    clf.fit(f_train, t_train)

    t_predict = clf.predict(f_test)

    dat = np.zeros((2, 2))
    for i in xrange(len(t_predict)):
        dat[t_test[i], t_predict[i]] += 1

    print 'Test results'
    print 'Predicted A | Predicted B'
    print dat

    t_train_predict = clf.predict(f_train)

    print 'Training results'
    print 'Predicted A | Predicted B'
    
    dat = np.zeros((2, 2))
    for i in xrange(len(t_train_predict)):
        dat[t_train[i], t_train_predict[i]] += 1

    print dat

if __name__ == '__main__':
    a_b_classify(2)
