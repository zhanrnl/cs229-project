from sklearn.decomposition import PCA
from metric_learn import *
from sklearn.neighbors import BallTree, DistanceMetric
import random
import sys
from multiprocessing import Pool

class TrialData(object):
    def __init__(self, pca_components=35, cross_validation_k=8, 
            correct_within_top_fraction=0.1, match_a_to_b=True):
        self.pca_components = pca_components
        self.cross_validation_k = cross_validation_k
        self.correct_within_top_fraction = correct_within_top_fraction
        self.match_a_to_b = match_a_to_b

    def split_data(self, all_pairs, i):
        num_pairs = len(all_pairs)
        i = float(i)
        start_split = int(i / self.cross_validation_k * num_pairs)
        end_split = int((i+1) / self.cross_validation_k * num_pairs)
        train = all_pairs[0:start_split] + all_pairs[end_split:]
        test = all_pairs[start_split:end_split]
        return train, test

    def fit_pca(self, train_pairs, test_pairs):
        train_pairs_flat = [item for subtuple in train_pairs for item in subtuple]
        test_pairs_flat = [item for subtuple in test_pairs for item in subtuple]
        
        pca = PCA(n_components = self.pca_components)
        pca.fit(train_pairs_flat)

        train_pairs_pca_flat = pca.transform(train_pairs_flat)
        test_pairs_pca_flat = pca.transform(test_pairs_flat)

        train_pairs_pca = list()
        test_pairs_pca = list()

        for i in xrange(0, len(train_pairs_pca_flat), 2):
            a = i 
            b = i + 1
            train_pairs_pca.append((train_pairs_pca_flat[a],
              train_pairs_pca_flat[b]))

        for i in xrange(0, len(test_pairs_pca_flat), 2):
            a = i 
            b = i + 1
            test_pairs_pca.append((test_pairs_pca_flat[a],
              test_pairs_pca_flat[b]))
        
        return train_pairs_pca, test_pairs_pca

    def get_full_metric(self, train_pairs):
        train_pairs_flat = [item for subtuple in train_pairs for item in subtuple]
        
        pca = PCA(n_components = self.pca_components)
        pca.fit(train_pairs_flat)

        train_pairs_pca_flat = pca.transform(train_pairs_flat)

        train_pairs_pca = list()

        for i in xrange(0, len(train_pairs_pca_flat), 2):
            a = i 
            b = i + 1
            train_pairs_pca.append((train_pairs_pca_flat[a],
              train_pairs_pca_flat[b]))
        
        ys = ys_from_pairs(train_pairs_pca)

        file_id = str(random.random())[2:]

        save_cvx_params(ys, file_id)
        run_cvx(file_id)
        M = load_cvx_result(file_id)

        dist = DistanceMetric.get_metric('mahalanobis', VI = M)

        return dist, M, pca

    def run_single_trial(self, train_pairs, test_pairs, train_tune_data, test_tune_data):
        print "Running PCA..."
        train_pairs_pca, test_pairs_pca = self.fit_pca(train_pairs, test_pairs)
        ys = ys_from_pairs(train_pairs_pca)

        file_id = str(random.random())[2:]

        save_cvx_params(ys, file_id)
        run_cvx(file_id)
        M = load_cvx_result(file_id)
        dist = DistanceMetric.get_metric('mahalanobis', VI = M)
        train_a_sections = [x[0] for x in train_pairs_pca]
        train_b_sections = [x[1] for x in train_pairs_pca]
        test_a_sections = [x[0] for x in test_pairs_pca]
        test_b_sections = [x[1] for x in test_pairs_pca]

        train_given_sections = train_a_sections
        train_to_match_sections = train_b_sections
        test_given_sections = test_a_sections
        test_to_match_sections = test_b_sections
        if self.match_a_to_b:
            train_given_sections = train_b_sections
            train_to_match_sections = train_a_sections
            test_given_sections = test_b_sections
            test_to_match_sections = test_a_sections

        print "Constructing BallTrees..."
        train_bt = BallTree(train_to_match_sections, metric=dist)
        test_bt = BallTree(test_to_match_sections, metric=dist)

        train_top_fraction = int(len(train_given_sections) * self.correct_within_top_fraction)
        test_top_fraction = int(len(test_given_sections) * self.correct_within_top_fraction)

        print "Querying the BallTrees..."
        train_result = train_bt.query(train_given_sections, train_top_fraction)
        test_result = test_bt.query(test_given_sections, test_top_fraction)

        print "Looking at correctness of results..."
        train_correct = sum([int(i in train_result[1][i]) for i in xrange(len(train_given_sections))])
        test_correct = sum([int(i in test_result[1][i]) for i in xrange(len(test_given_sections))])

        print "Finding indices of correct matches..."
        test_result_full = test_bt.query(test_given_sections, len(test_given_sections))
        def default_index(lst, i):
          ind = -1
          try:
            ind = lst.index(i)
          except:
            pass
          return ind
        test_indices = [default_index(list(test_result_full[1][i]), i) for i in xrange(len(test_given_sections))]
        test_indices = [x for x in test_indices if x != -1]

        with open("successful_tunes_{}".format(file_id), 'w') as successful_tunes_f:
          for i, index in enumerate(test_indices):
            if index == 0:
              successful_tunes_f.write(str(test_tune_data[i]) + '\n\n')

        return [[train_correct, len(train_given_sections)],
            [test_correct, len(test_given_sections)]], test_indices

    def print_results(self, results, indices, outfile=sys.stdout, indices_outfile=sys.stdout):
        ((train_correct, num_train),
            (test_correct, num_test)) = results
        outfile.write( """Ran {}-fold cross validation with {} PCA'd components, matching {} sections
to {} sections, and checking whether the corresponding section was in the
top {}% closest by the learned metric.

"""\
            .format(self.cross_validation_k, self.pca_components,
              'A' if self.match_a_to_b else 'B', 'B' if self.match_a_to_b else 'A',
              self.correct_within_top_fraction * 100))
        outfile.write( """ON THE TEST SET:
{} correct by learned metric, {} total

"""\
            .format(test_correct, num_test))
        outfile.write( """ON THE TRAINING SET:
{} correct by learned metric, {} total

"""\
            .format(train_correct, num_train))
        outfile.write("\n")
        indices_outfile.write(str(indices))
        indices_outfile.write("\n")

    def run_trial(self, all_pairs, tune_data, outfile=sys.stdout, indices_outfile=sys.stdout):
        pool = Pool()
        #print len(all_pairs), len(tune_data)
        map_result = pool.map_async(single_trial, 
            [(i, self, all_pairs, tune_data) for i in xrange(self.cross_validation_k)], 1)
        pool.close()
        pool.join()
        results_list = map_result.get()
        results = [[0,0], [0,0]]
        all_indices = []
        for result, indices in results_list:
          all_indices.extend(indices)
          for j in xrange(2):
            for k in xrange(2):
              results[j][k] += result[j][k]
        self.print_results(results, all_indices, outfile, indices_outfile)

def single_trial((i, trial_data, all_pairs, tune_data)):
  print "Running the {}th cross validation trial...".format(i+1)
  train, test = trial_data.split_data(all_pairs, i)
  train_tune_data, test_tune_data = trial_data.split_data(tune_data, i)
  return trial_data.run_single_trial(train, test, train_tune_data, test_tune_data)

if __name__ == '__main__':
    pairs, tune_data = build_a_b_pairs_vector(2, 6)

    with open('trial_data.log_norhythm_btoa', 'w') as outfile:
      with open('trial_data_indices_norhythm_btoa', 'w') as indices_outfile:
        trial_data = TrialData(pca_components=35)
        trial_data.run_trial(pairs, tune_data, outfile, indices_outfile)
