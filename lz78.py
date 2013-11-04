def k_est(s):
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

def k_concat(x, y):
    return float(k_est(x + y) + k_est(y + x)) / 2

def k_cond(x, y):
    return k_concat(x, y) - k_est(y)

def metric_d1(x, y):
    assert(len(x) > 0 or len(y) > 0)
    return float(k_cond(x, y) + k_cond(y, x)) / k_concat(x, y)

def metric_d2(x, y):
    assert(len(x) > 0 or len(y) > 0)
    return float(max(k_cond(x, y), k_cond(y, x))) / max(k_est(x), k_est(y))
