from itertools import islice
import numpy as np

one_grams = [str(x) for x in xrange(1, 8)]

def build_n_grams_helper(n_gram_list_list = None):
    ''' 
    Given a list where the nth element is the list of n-grams, returns
    a copy of this list with an (n+1)th element that is the list of (n+1)-grams
    '''

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
    Given a tuple of pitches, returns string of 1..7 to be used as a key
    in the index dictionary
    '''
    c_list = [ord(x) - ord('a') + 1 for x in gram_tuple]
    return ''.join([str(x) for x in c_list if (x >= 1 and x <= 7)])

# LENNART .pitch SOMETIMES RETURNS NOTES INSTEAD OF STRINGS WHY?!?!?!
# (hence the str(x.pitch))
def features_from_list_of_bars(list_of_bars, n_gram_dict, n = 3):
    ''' 
    Given a list of bars (i.e. the 'parsed' element of a Tune) return the
    associated feature vector for all k-grams, k <= n
    '''

    pitch_str = ''.join([str(x.pitch) for b in list_of_bars for x in b]).lower()

    features = np.zeros(len(n_gram_dict))

    for k in xrange(1, n + 1):
        for gram in window(pitch_str, k):
            gram_str = gram_str_from_tuple(gram)
            if len(gram_str) > 0:
                features[n_gram_dict[gram_str]] += 1

    return features

type_dict = {'jig': 0, 'reel': 1, 'slip jig': 2, 'hornpipe': 3, 'polka': 4, 'slide': 5, 'waltz': 6, 'barndance': 7, 'strathspey': 8, 'three-two': 9, 'mazurka': 10}

def build_all_feature_vectors(n = 3):
    '''
    Builds all feature vectors from the whole list of parsed songs
    '''
    
    import cPickle

    f_vecs = list()
    types = list()

    d = build_feature_index_map(n)

    # 6 is hardcoded number of chunks of the parsed list
    for i in xrange(6):
        tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'.format(i), 'rb'))
        
        for tune in tunes:
            if 'parsed' in tune:
                f_vecs.append(features_from_list_of_bars(tune['parsed'], d, n))
                types.append(type_dict[tune['type']])

    return f_vecs, types

def train_test_split(f_vecs, types):
    n_train = int(round(0.7 * len(f_vecs)))

    f_train = f_vecs[:n_train]
    t_train = types[:n_train]

    f_test = f_vecs[n_train:]
    t_test = types[n_train:]

    return f_train, t_train, f_test, t_test

if __name__ == '__main__':
    from sklearn import svm
    f_vecs, types = build_all_feature_vectors()

    f_train, t_train, f_test, t_test = train_test_split(f_vecs, types)
    
    lin_clf = svm.LinearSVC()
    lin_clf.fit(f_train, t_train)
    
    t_predict = lin_clf.predict(f_test)

    type_errors = np.zeros(11, dtype='int32')
    type_nums = np.zeros(11, dtype='int32')

    for i in range(len(t_test)):
        type_nums[t_test[i]] += 1
        if t_test[i] != t_predict[i]:
            type_errors[t_test[i]] += 1

    frac_total_error = sum(type_errors) / float(len(t_test))
    frac_error = np.zeros(11)
    for i in range(11):
        frac_error[i] = type_errors[i] / float(type_nums[i])
    
    print frac_total_error
    print frac_error
