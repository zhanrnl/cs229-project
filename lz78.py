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
