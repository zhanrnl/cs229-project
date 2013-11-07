def k_est(s):
    ''' estimate length of LZ78 compression '''
    d = set()
    w = ''
    Kest = 0

    for c in s:
        if w + c in d:
            w = w + c
        else:
            Kest += 1
            d.add(w + c)
            w = ''

    if w != '':
        Kest += 1

    return Kest

def ncd(x, y):
    ''' "Normalized Compression Distance" -- see homepages.cwi.nl/~paulv/papers/cluster.pdf '''
    c_x = k_est(x)
    c_y = k_est(y)
    c_xy = (k_est(x + y) + k_est(y + x)) * 0.5

    return (c_xy - min(c_x, c_y)) / max(c_x, c_y)
