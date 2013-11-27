from n_grams import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
import cPickle
import time

def build_a_b_features_labels(n = 3, num_blocks = 6):
    ''' 
    Reads in all tunes and creates feature vectors for each
    We hard code 0 as A section, 1 as B section 
    '''
    assert(num_blocks >= 1 and num_blocks <= 6)

    f_vecs = list()
    types = list()

    d = build_feature_index_map(n)

    for i in xrange(num_blocks):
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


def a_b_classify(n = 2, num_blocks = 6):
    '''
    Uses an SVM to classify A and B sections based on the feature vectors
    built above, and returns some statistical results
    '''
    print '{0}: Building feature vectors'.format(time.ctime())
    f_vecs, types = build_a_b_features_labels(n = n, num_blocks = num_blocks)

    f_train, t_train, f_test, t_test = train_test_split(f_vecs, types, fraction = 0.9)

    print '{0}: Training the SVM'.format(time.ctime())
    clf = svm.SVC()
    clf.fit(f_train, t_train)

    t_predict = clf.predict(f_test)
    t_train_predict = clf.predict(f_train)
    
    print 'Confusion matrix is built so that C_ij is the number of observations known to be in group i but predicted to be in group j. In this case, group 0 corresponds to A sections and group 1 corresponds to B sections.'
    
    print 'Confusion matrix on test data:'
    print confusion_matrix(t_test, t_predict)

    print 'Confusion matrix on training data:'
    print confusion_matrix(t_train, t_train_predict)

if __name__ == '__main__':
    a_b_classify(n = 2, num_blocks = 6)
