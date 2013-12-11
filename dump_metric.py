from trial_data import *
from sklearn.neighbors import DistanceMetric
import cPickle

if __name__ == '__main__':
    pairs, tune_data = build_a_b_pairs_vector(2, 6)
    trial_data = TrialData(pca_components = 35, cross_validation_k = 1)
    dist, M, pca = trial_data.get_full_metric(pairs)
    cPickle.dump(dist, open('metric_dump', 'wb'))
    cPickle.dump(M, open('M_full', 'wb'))
    cPickle.dump(pca, open('pca_full', 'wb'))
