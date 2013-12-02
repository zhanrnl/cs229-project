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
    Generates a dictionary with keys that are n-grams of notes (A..g),
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

def features_from_list_of_bars(list_of_bars, n_gram_dict, n = 3):
    ''' 
    Given a list of bars (i.e. the 'parsed' element of a Tune) return the
    associated feature vector for all k-grams, k <= n
    '''
    pitch_str = ''.join([str(x.pitch) for b in list_of_bars for x in b])

    # update this later when we add accidentals
    pitch_str = [x for x in pitch_str if x in ''.join(feature_list)]

    features = np.zeros(len(n_gram_dict))

    for k in xrange(1, n + 1):
        for gram in window(pitch_str, k):
            features[n_gram_dict[gram]] += 1

    return features

def features_multibar_split(bars, n_gram_dict, n=3):
  notes = [no for b in bars for no in b]
  length = int(total_length([notes]))
  num_slices = 0
  f_vec = []
  eighth_slice = []
  for no in notes:
    if num_slices == 8:
      break
    eighth_slice.append(no)
    if total_length([eighth_slice]) >= (length / 8):
      f_vec.extend(features_from_list_of_bars([eighth_slice], n_gram_dict, n))
      eighth_slice = []
      num_slices += 1
  for i in range(8 - num_slices):
    f_vec.extend(features_from_list_of_bars([], n_gram_dict, n))
  return f_vec

  #slice_len = int(len(bars) / 8)
  #for i in range(8):
    #f_vec.extend(features_from_list_of_bars(
      #bars[(i*slice_len) : ((i+1)*slice_len)], n_gram_dict, n))
  #return f_vec
