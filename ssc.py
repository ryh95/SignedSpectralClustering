import numpy as np


def find_r(x, z):
    u, _, v = np.linalg.svd(z.H * x)
    return u*v


def make_ortho_rows(a):
    zeroes = np.sum(np.abs(a.H), axis=0)
    a = a[zeroes.nonzero(), :]
    n = a.shape[0]
    k = a.shape[1]
    R = np.eye(k)
    if n < k:
        fail = 1
        return [R, fail]
    idx = 1
    R[:, 0] = a[idx, :].H
    a2 = np.zeros((n-1, k))
    a2[0:idx-2, :] = a[0:idx - 2, :]
    a2[idx:n-2, :] = a[idx:n - 1, :]
    c = np.zeros((n-1, 1))
    for i in range(1, k):
        c += np.abs(a2 * R[:, i-2])
        I = np.amin(c)
        R[:, i] = a2[I, :].H
        a3 = np.union1d(a2[range(0, I-2), ], a2[range(I, n-i), ])
        # a3 = a2[range(0, I-2) + range(I, n-i), ]
        c2 = c[range(0, I-2) + range(I, n-i), 1]
        c = c2
        a2 = a3
    nrc = np.zeros((k, k))
    rr = R.H.dot(R)
    for i in range(0, k):
        nrc[i, i] = np.reciprocal(np.sqrt(rr[i, i]))
    return R.dot(nrc.H)


def colpos(z):
    n = z.shape[0]
    k = z.shape[1]
    rr = np.eye(k)
    col_sum = np.ones((1, n)) * z
    z2 = z
    for i in range(0, k):
        if col_sum[0, i] < 0:
            z2[:, i] = -z[:, i]
            rr[i, i] = -1
    return [z2, rr]


def init_R2(z1, z2, k, a):
    r2, r2z = make_ortho_rows(z2)
    r2a, r2za = make_ortho_rows(z1)
    r3 = np.eye(k)

    # Finds Xc1 using Z1 = R3 = I
    Xc1, RR1 = find_hard_clusters(z1, r3, a)
    N1 = np.sqrt(np.trace((Xc1 - z1 * RR1).H * (Xc1 - z1 * RR1)))

    # Finds Xc1 using Z2 = Z1*R1 and R3 = I
    Xc3, RR3 = find_hard_clusters(z2, r3, a)
    N3 = np.sqrt(np.trace((Xc3 - z2 * RR3).H * (Xc3 - z2 * RR3)))
    sw = np.amin([N1, N3])
    if sw == 1:
        rc = RR1
        xc = Xc1
    else:
        rc = RR3
        xc = Xc3
    if r2z == 1 or r2za == 1:
        sw = 1
    return [rc, sw, xc]

    # # Finds Xc1 using Z1 and R2a
    # Xc2, RR2 = find_hard_clusters(z1, r2a, a)
    # N2 = np.sqrt(np.trace((Xc2 - z1 * r2a * RR1).H * (Xc1 - z1 * r2a * RR1)))
    #
    # # Finds Xc1 using Z2 = Z1*R1 and R3 = I
    # Xc3, RR3 = find_hard_clusters(z2, r3, a)
    # N3 = np.sqrt(np.trace((Xc3 - z2 * RR3).H * (Xc3 - z2 * RR3)))
    #
    # # Finds Xc1 using Z2 and R2
    # Xc4, RR4 = find_hard_clusters(z2, r2, a)
    # N4 = np.sqrt(np.trace((Xc4 - z2 * r2 * RR4).H * (Xc4 - z2 * r2 * RR4)))
    #
    # sw = np.amin([N1, N2, N3, N4])
    # if sw == 1:
    #     rc = RR1
    #     xc = Xc1
    # elif sw == 2:
    #     rc = RR2
    #     xc = Xc2
    # elif sw == 3:
    #     rc = RR3
    #     xc = Xc3
    # else:
    #     rc = RR4
    #     xc = Xc4
    # if r2z == 1 or r2za == 1:
    #     sw = 1
    # return [rc, sw, xc]


def find_soft_clusters(z, r, a):
    zr1 = z * r
    k = zr1.shape[1]
    zr2, rn = colpos(zr1)
    for st in range(0, 2):
        zr = zr1 if st == 0 else zr2
        zr = zr.clip(min=0)
        n = zr.shape[0]
        x1 = zr;
        for i in range(0, n):
            x1[i, :] = x1[i, :]/np.sum(x1[i, :])
        if st == 0:
            xx1 = a*x1
        else:
            xx2 = a*x1

    n1 = np.sqrt(np.trace((xx1 - zr1).H * (xx1 - zr1)))
    n2 = np.sqrt(np.trace((xx2 - zr2).H * (xx2 - zr2)))
    x = xx2 if n1 > n2 else xx1
    return [x, rn]


def find_flex_clusters(z, r, a):
    zr1 = z * r
    k = zr1.shape[1]
    zr2, rn = colpos(zr1)
    for st in range(0, 2):
        zr = zr1 if st == 0 else zr2
        n = zr.shape[0]
        x1 = np.zeros((n, k))
        J = np.argmax(zr.H, axis=0)
        for i in range(0, n):
            x1[i, J[0, i]] = 1

        if st == 1:
            xx1 = a*x1
        else:
            xx2 = a*x1
            
    n1 = np.sqrt(np.trace((xx1 - zr1).H * (xx1 - zr1)))
    n2 = np.sqrt(np.trace((xx2 - zr2).H * (xx2 - zr2)))
    x = xx2 if n1 > n2 else xx1
    return [x, rn]


def find_hard_clusters(z, r, a):
    zr1 = z * r
    k = zr1.shape[1]
    zr2, rn = colpos(zr1)
    for st in range(0, 2):
        zr = zr1 if st == 0 else zr2
        n = zr.shape[0]
        x1 = np.zeros((n, k))
        J = np.argmax(zr.H, axis=0)
        for i in range(0, n):
            x1[i, J[0, i]] = 1

        x1_sum = np.sum(x1, axis=0)
        row_idx = np.argmax(x1, axis=0)
        sum_idx = np.argmax(x1_sum, axis=0)
        for j in range(0, int(k)):
            if x1_sum[j] == 0:
                x1[row_idx[sum_idx]][sum_idx] = 0
                x1[row_idx[sum_idx]][j] = 1
                x1_sum = sum(x1)
                row_idx = np.argmax(x1, axis=0)
                sum_idx = np.argmax(x1_sum, axis=0)
        if st == 0:
            xx1 = a*x1
        else:
            xx2 = a*x1

    n1 = np.sqrt(np.trace((xx1 - zr1).H * (xx1-zr1)))
    n2 = np.sqrt(np.trace((xx2 - zr2).H * (xx2 - zr2)))
    x = xx2 if n1 > n2 else xx1
    return [x, rn]


def sncut(w, n_clusters=4, threshold=1e-14, max_iters=50):
    n_vert = w.shape[0]

    # Compute signed normalized Laplacian
    deg_sums = np.absolute(w).sum(axis=0)
    deg_mat = np.diagflat(deg_sums)
    deg_sums_sqrt = np.sqrt(deg_sums)
    deg_inv = np.diagflat(np.reciprocal(deg_sums_sqrt , where=deg_sums_sqrt != 0))
    lap = deg_mat-w
    lap_sym = deg_inv.dot(lap).dot(deg_inv)

    # Initialize U to be vector of the K smallest eigenvalues of l_sym
    u, _, _ = np.linalg.svd(2*np.eye(n_vert) - lap_sym)
    u_k = u[:, 0:n_clusters]

    # Find Z
    z1 = deg_inv.dot(u_k)
    nz = np.sqrt(np.trace(z1.H.dot(z1)))
    z1 *= (100/nz)

    _, v = np.linalg.eig(z1.H.dot(z1))
    z2 = z1.dot(v)

    a = 100/np.sqrt(n_vert)
    nz1 = np.sqrt(np.trace(z1.H.dot(z1)))
    nz2 = np.sqrt(np.trace(z2.H.dot(z2)))
    z1 *= (100/nz1)
    z2 *= (100/nz2)

    rc, sw, xc = init_R2(z1, z2, n_clusters, a)
    z = z1 if sw == 1 or sw == 2 else z2

    ex = np.sqrt(np.trace((xc - z.dot(rc)).H.dot(xc - z.dot(rc))))
    rc = find_r(xc, z)

    ern = np.sqrt(np.trace((xc - z.dot(rc)).H.dot(xc - z.dot(rc))))
    er = ern + 1
    convlist = np.zeros(2 * max_iters)
    convlist[1] = ex
    convlist[2] = ern
    xc0 = xc
    rc0 = rc

    step = 1
    while ern < er and step <= max_iters:
        step += 1
        xc, _ = find_hard_clusters(z, rc, a)
        diff_xc = (xc - xc0)/a
        ndxc = np.trace(diff_xc.conj().T.dot(diff_xc))/2
        x_change = 1 if ndxc < threshold else 0
        ex = np.sqrt(np.trace((xc - z.dot(rc)).H.dot(xc - z.dot(rc))))
        convlist[2*step-1] = ex
        if x_change == 1:
            convlist[2*step] = ern
            ern = er
        else:
            rc = find_r(xc, z)
            er = ern
            ern = np.sqrt(np.trace((xc - z.dot(rc)).H.dot(xc - z.dot(rc))))
            convlist[2 * step] = ern
            if er < ern:
                xc = xc0
                rc = rc0
            else:
                xc0 = xc
                rc0 = rc

    xxc = (1/a)*xc
    indices = np.argmax(xxc.conj().T, axis=0)
    return [indices.conj().T, xxc]


if __name__ == "__main__":
    labels = ['Kiana', 'Trang', 'Olivia', 'Sammy', 'Philippe',
              'Ryan', 'Alex', 'Wesley', 'Stefi', 'Shane', 'Harrison',
              'Michael']
    w = np.matrix('\
            0 -0.5 0.9 0.9 0.5 0.4 0.2 0.3 0.1 0.2 0.3 0.4; \
            -0.5 0 -0.8 -0.8 -0.1 0.8 0.1 0.1 0.1 0.1 0.3 0.1; \
            0.9 -0.8 0 0.9 1.0 0.3 0.4 0.4 0.1 0.5 0.2 0.2; \
            0.9 -0.8 0.9 0 0.4 0.7 0.1 0.1 0.1 0.3 0.1 0.4; \
            0.5 -0.1 1.0 0.4 0 0.8 0.5 0.3 -0.1 0.1 0.5 0.1; \
            0.4 0.8 0.3 0.7 0.8 0 -0.2 0.1 0.1 0.1 0.4 0.3; \
            0.2 0.1 0.4 0.1 0.5 -0.2 0 0.5 -0.1 -0.1 0.4 0.1; \
            0.3 0.1 0.4 0.1 0.3 0.1 0.5 0 0.3 0.3 0.3 0.1; \
            0.1 0.1 0.1 0.1 -0.1 0.1 -0.1 0.3 0 0.2 0.1 -0.1; \
            0.2 0.1 0.5 0.3 0.1 0.1 -0.1 0.3 0.2 0 0.3 0.2; \
            0.3 0.3 0.2 0.1 0.5 0.4 0.4 0.3 0.1 0.3 0 -0.2; \
            0.4 0.1 0.2 0.4 0.1 0.3 0.1 0.1 -0.1 0.2 -0.2 0')
    k = 6
    idx, XXc = sncut(w, n_clusters=k)
    for dim in range(0, k):
        print("\n")
        print("Group %d" % dim)
        for i in range(0, w.shape[0]):
            if XXc[i, dim] > 0:
                print(labels[i])
