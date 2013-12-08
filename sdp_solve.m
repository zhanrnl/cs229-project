load('cmat.mat')
n = size(cmat, 1);

assert(n <= 36);

cvec = reshape(cmat, n^2, 1);

cvx_begin sdp
    variable M(n,n) symmetric
    minimize(cvec' * reshape(M, n^2, 1))
    subject to
        M >= 0
        det_rootn(M) >= 1
cvx_end

save('result', 'M');
