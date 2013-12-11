from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA
from sessionparse import *
import cPickle
from n_grams import *
import random
import copy
import time

def read_metric():
    M = cPickle.load(open('M_full', 'rb'))

    def metric(x, y):
        return np.sqrt((y - x).T.dot(M).dot(y - x))

    return metric

n = 2
d = build_feature_index_map(n)
metric = read_metric()
pca = cPickle.load(open('pca_full', 'rb'))

def load_pickled((i, d, n)):
    f_vecs = []
    tunes = cPickle.load(open('thesession-data/cpickled_parsed_{}'\
        .format(i), 'rb'))
    print '{}: pickle loaded file {}'.format(time.ctime(), i)

    num_split = 0
    num_not_split = 0
    num_not_parsed = 0
    for tune in tunes:
        if 'parsed' not in tune:
          num_not_parsed += 1
          continue

        try:
          a, b = ab_split(tune)
          num_split += 1
        except ValueError as e:
          num_not_split += 1
          continue

        f_vecs.append(double_feature_vec(a, d, n))
        f_vecs.append(double_feature_vec(b, d, n))
        #f_vecs.append(features_multibar_split(a, d, n))
        #f_vecs.append(features_multibar_split(b, d, n))
        #f_vecs.append(features_from_list_of_bars(a, d, n))
        #f_vecs.append(features_from_list_of_bars(b, d, n))

    print '{} tunes successfully split, {} did not split, {} '\
        'didn\'t parse at all, {} total'.format(num_split,
            num_not_split, num_not_parsed, num_not_parsed + num_split
            + num_not_split)
    return f_vecs

def save_pcad_dataset():
    pool = Pool()

    n = 2
    d = build_feature_index_map(n)
    mapresult = pool.map_async(load_pickled, [(i, d, n) for i in range(6)], 1)
    pool.close()
    pool.join()
    results = mapresult.get()

    f_vecs = []
    for f_vec_file in results:
      for f_vec in f_vec_file:
        f_vecs.append(f_vec)
    large_pca = PCA(n_components=100)
    import pdb; pdb.set_trace()
    large_pca.fit(f_vecs)

    for i, f_vec in enumerate(f_vecs):
      f_vecs[i] = large_pca.transform(f_vec)

    with open('pcad_dataset', 'w') as f:
      cPickle.dump(f_vecs, f)


def get_cooleys():
    tunes = cPickle.load(open('thesession-data/cpickled_parsed_0', 'rb'))
    return tunes[3945]

def random_pitch():
    poss_notes = list('ABCDEFGabcdefg')
    return poss_notes[random.randint(0, len(poss_notes) - 1)]

def random_alteration(bars):
    bars_copy = copy.deepcopy(bars)

    num_notes_to_change = 2

    bars_to_change = random.sample(xrange(len(bars_copy)), 3)

    for i in bars_to_change:
        bar = bars_copy[i]
        notes = bar.notes

        this_num_notes_to_change = min(len(notes), num_notes_to_change)
        if this_num_notes_to_change <= 0:
            continue

        inx_to_change = random.randint(0, len(notes) - this_num_notes_to_change)
    
        for j in xrange(inx_to_change, inx_to_change + this_num_notes_to_change):
            notes[j].pitch = random_pitch()

    return bars_copy

def total_random(bars):
    bars_copy = copy.deepcopy(bars)

    for i in xrange(len(bars_copy)):
        bar = bars_copy[i]

        for j in xrange(len(bar.notes)):
            bar.notes[j].pitch = random_pitch()

    return bars_copy

def get_fvec(bars):
    return pca.transform(double_feature_vec(bars, d, n)).reshape((35, 1))

def iterative_step(a_fvec, curr_b, num_candidates = 10):
    candidates = [random_alteration(curr_b) for i in xrange(num_candidates)]
    candidates.append(curr_b)
    candidate_fvecs = [get_fvec(candidate) for candidate in candidates]

    dists = [metric(a_fvec, b_fvec) for b_fvec in candidate_fvecs]
    min_dist = min(dists)

    best_b = candidates[dists.index(min_dist)]

    return best_b, min_dist
    

def compose(tune, iters = 100, n_candidates = 10, seed = None):
    mode = tune['mode']

    a, b = ab_split(tune)

    a_fvec = get_fvec(a)

    if seed == None:
        seed = total_random(a)
   
    curr_b = seed

    for i in xrange(iters):
        curr_b, dist = iterative_step(a_fvec, curr_b, n_candidates)
        print 'iteration #{}/{}'.format(i+1, iters), 'dist: {}'.format(float(dist))

    return curr_b, seed

def main():
    tune = get_cooleys()
    
    b, orig_rand = compose(tune, 500, 50)

    cPickle.dump((tune, b, orig_rand), open('tune_b_seed','wb'))

if __name__ == '__main__':
  pass
    #main()
