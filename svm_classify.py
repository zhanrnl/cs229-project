from sklearn.decomposition import PCA
from n_grams import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
import cPickle
import time
from multiprocessing import Pool
import itertools
from collections import defaultdict

debug = False

def get_durations():
  d = defaultdict(int)
  for i in range(6):
    tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'\
        .format(i), 'rb'))
    print '{}: pickle loaded file {}'.format(time.ctime(), i)
    for tune in tunes:
      if 'parsed' in tune:
        for bar in tune['parsed']:
          for note in bar.notes:
            d[note.dur] += 1
  return d

def load_pickled((i, d, n)):
    f_vecs = []
    types = []
    tunes = cPickle.load(open('thesession-data/cpickled_parsed_{0}'\
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
          if debug:
            print e
          num_not_split += 1
          continue

        #f_vecs.append(double_feature_vec(a, d, n))
        #f_vecs.append(double_feature_vec(b, d, n))
        f_vecs.append(features_multibar_split(a, d, n))
        f_vecs.append(features_multibar_split(b, d, n))
        #f_vecs.append(features_from_list_of_bars(a, d, n))
        #f_vecs.append(features_from_list_of_bars(b, d, n))
        types.append(0)
        types.append(1)

    print '{} tunes successfully split, {} did not split, {} '\
        'didn\'t parse at all, {} total'.format(num_split,
            num_not_split, num_not_parsed, num_not_parsed + num_split
            + num_not_split)
    return (f_vecs, types)

def build_a_b_features_labels(n = 3, num_blocks = 6):
    ''' 
    Reads in all tunes and creates feature vectors for each
    We hard code 0 as A section, 1 as B section 
    '''
    assert(num_blocks >= 1 and num_blocks <= 6)

    d = build_feature_index_map(n)

    pool = Pool()

    mapresult = pool.map_async(load_pickled, [(i, d, n) for i in range(num_blocks)], 1)
    pool.close()
    pool.join()
    results = mapresult.get()
    f_vecs = []
    types = []
    for f, t in results:
      f_vecs.extend(f)
      types.extend(t)

    return f_vecs, types
                

def train_test_split(f_vecs, types, fraction = 0.7):
    '''
    Splits all data into a training and test set, where the training set
    is {fraction} of our data
    '''
    assert(fraction > 0 and fraction < 1)

    n_train = int(round(fraction * len(f_vecs)))

    f_train = f_vecs[:n_train]
    t_train = types[:n_train]

    f_test = f_vecs[n_train:]
    t_test = types[n_train:]

    t = (f_train, t_train, f_test, t_test)
    
    return tuple([np.array(x) for x in t])


def a_b_classify_pca((f_train, t_train, f_test, t_test, n_components)):
    '''
    Uses an SVM to classify A and B sections based on the feature vectors
    built above, and returns some statistical results
    '''
    print '{}: Starting PCA with {} components (this could take a while...)'.format(time.ctime(), n_components)
    pca = PCA(n_components = n_components)
    pca.fit(f_train)
    f_train_pca = list(pca.transform(f_train))
    f_test_pca = list(pca.transform(f_test))

    print '{0}: Training the SVM'.format(time.ctime())
    clf = svm.SVC()
    clf.fit(f_train_pca, t_train)

    print '{0}: Classifying using SVM'.format(time.ctime())
    t_predict = clf.predict(f_test_pca)
    t_train_predict = clf.predict(f_train_pca)
    
    print 'Confusion matrix is built so that C_ij is the number of observations known to be in group i but predicted to be in group j. In this case, group 0 corresponds to A sections and group 1 corresponds to B sections.'
    
    print 'Confusion matrix on test data:'
    test_confuse = confusion_matrix(t_test, t_predict)
    print test_confuse

    print 'Confusion matrix on training data:'
    train_confuse = confusion_matrix(t_train, t_train_predict)
    print train_confuse
    return train_confuse, test_confuse
    
def split_data(all_pairs, i, k):
    num_pairs = len(all_pairs)
    i = float(i)
    start_split = int(i / k * num_pairs)
    end_split = int((i+1) / k * num_pairs)
    train = all_pairs[0:start_split] + all_pairs[end_split:]
    test = all_pairs[start_split:end_split]
    return train, test

if __name__ == '__main__':
  print '{0}: Building feature vectors'.format(time.ctime())
  f_vecs, types = build_a_b_features_labels(n = 2, num_blocks = 6)

  for n_components in [5, 10, 20, 30, 40, 60]:
    print "Classifying with {} components in PCA".format(n_components)
    splits = []
    k = 8
    for i in range(k):
      f_train, f_test = split_data(f_vecs, i, k)
      t_train, t_test = split_data(types, i, k)
      splits.append((f_train, t_train, f_test, t_test, n_components))
    pool = Pool()
    map_result = pool.map_async(a_b_classify_pca, splits, 1)
    pool.close()
    pool.join()
    result = map_result.get()

    train_confuse_sum = np.zeros((2, 2))
    test_confuse_sum = np.zeros((2, 2))
    for train_confuse, test_confuse in result:
      train_confuse_sum = train_confuse_sum + train_confuse
      test_confuse_sum = test_confuse_sum + test_confuse
    with open("svm_result_norhythm_{}".format(n_components), 'w') as f:
      f.write(str(train_confuse_sum))
      f.write('\n')
      f.write(str(test_confuse_sum))

