
import numpy as np
from tensorly.tenalg import khatri_rao
from sporco.linalg import fftn, ifftn
from scipy.sparse import csr_matrix, hstack, kron, identity
import time


def T_ConvFISTA_precompute(Gram, Achapy, Z_init, L, lbd, beta, maxit, tol=1e-5, verbose=False):
    """ Minimization of the sub-block of Z

        Gram: Gram matrix
        Achapy: vector A * y
        Z_init: initialization
        L: Lipschitz constant
        lbd, beta: hyerparameters

    """

    K, N_i, R = Z_init.shape

    pobj = []
    time0 = time.time()

    Zpred = Z_init.copy()
    Xfista = Z_init.copy()
    t = 1
    ite = 0.
    tol_it = tol + 1.
    while (tol_it > tol) and (ite < maxit):
        if verbose:
            print(ite)

        Zpred_old = Zpred.copy()

        # DFT of the activations
        Xfistachap = fftn(Xfista, axes=[1])

        #  Vectorization
        xfistachap = np.reshape(Xfistachap, (K, N_i * R), order='F').ravel()

        # Computation of the gradient
        gradf = (Gram.dot(xfistachap) - Achapy) + 2 * beta * xfistachap

        # Descent step
        xfistachap = np.array(xfistachap - gradf / L)

        # Matricization
        Xfistachap = xfistachap.reshape(
            K, N_i * R).reshape((K, N_i, R), order='F')

        # IDFT of the activations
        Xfista = np.real(ifftn(Xfistachap, axes=[1]))

        # Soft-thresholding
        Zpred = np.sign(Xfista) * np.fmax(abs(Xfista) - lbd / L, 0.)

        # Nesterov Momentum
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        Xfista = Zpred + ((t0 - 1.) / t) * (Zpred - Zpred_old)

        # Stopping criterion
        tol_it = np.max(abs(Zpred_old - Zpred))
        this_pobj = tol_it.copy()

        ite += 1
        pobj.append((time.time() - time0, this_pobj))

    print('last iteration:', ite)
    times, pobj = map(np.array, zip(*pobj))
    return Zpred, times, pobj


def square_mat_csr(X):
    squared_X = X.copy()
    # now square the data in squared_X
    squared_X.data *= squared_X.data.conj()
    # and delete the squared_X:
    return(np.real(squared_X.sum()))


def f_Achap(M, Zchap, ZZchap, dchap, N, K):
    """ Computation of the matrix (A 'kron' I) """

    L = 0.  # norm of the matrix
    Achap = []
    for k in range(K):
        Bchap_k = kron(csr_matrix(khatri_rao(
            (Zchap[k], ZZchap[k]))).dot(M[k]), identity(N))
        Asparse = csr_matrix.multiply(csr_matrix(
            dchap[k][:, None]), Bchap_k)  #  fast diag product
        Achap.append(Asparse)

        L += square_mat_csr(Asparse)
    return hstack(Achap), L


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def consistency_on_rank(Z1, Z2, Z3, thr=0.):
    K, _, R = Z1.shape
    Z1_copy = Z1.copy()
    Z2_copy = Z2.copy()
    Z3_copy = Z3.copy()
    for k in range(K):
        for r in range(R):
            if np.linalg.norm(abs(Z1_copy)[k, :, r]) <= thr:
                Z2_copy[k, :, r] *= 0.
                Z3_copy[k, :, r] *= 0.

            if np.linalg.norm(abs(Z2_copy)[k, :, r]) <= thr:
                Z1_copy[k, :, r] *= 0.
                Z3_copy[k, :, r] *= 0.

            if np.linalg.norm(abs(Z3_copy)[k, :, r]) <= thr:
                Z1_copy[k, :, r] *= 0.
                Z2_copy[k, :, r] *= 0.
    return Z1_copy, Z2_copy, Z3_copy


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Specific ordering of the indexes during the vectorization


def unfold(tensor, mode, order='F'):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order=order)


def ravel(X, order='F'):
    return np.ravel(X, order=order)
