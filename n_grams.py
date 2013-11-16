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
