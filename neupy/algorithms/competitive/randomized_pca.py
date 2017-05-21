import numpy as np
from scipy import linalg


def svd_flip(u, v):
    """
    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the
    loadings in the columns in u that are largest in absolute
    value are always positive.

    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    Notes
    -----
    scikit-learn implementation

    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions
    as the input.
    """
    max_abs_rows = np.argmax(np.abs(v), axis=1)
    signs = np.sign(v[range(v.shape[0]), max_abs_rows])

    u *= signs
    v *= signs[:, np.newaxis]

    return u, v


def randomized_range_finder(A, size, n_iter):
    """
    Computes an orthonormal matrix whose range
    approximates the range of A.

    Parameters
    ----------
    A: 2D array
        The input data matrix

    size: integer
        Size of the return array

    n_iter: integer
        Number of power iterations used to stabilize the result

    Returns
    -------
    Q: 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----
    scikit-learn implementation
    """
    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = np.random.normal(size=(A.shape[1], size))

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        Q, _ = linalg.lu(np.dot(A, Q), permute_l=True)
        Q, _ = linalg.lu(np.dot(A.T, Q), permute_l=True)

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(np.dot(A, Q), mode='economic')
    return Q


def randomized_svd(M, n_components, n_oversamples=10):
    """
    Computes a truncated randomized SVD.

    Parameters
    ----------
    M: ndarray or sparse matrix
        Matrix to decompose

    n_components: int
        Number of singular values and vectors to extract.

    n_oversamples: int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    Notes
    -----
    scikit-learn implementation
    """
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape
    n_iter = 7 if n_components < .1 * min(M.shape) else 4

    Q = randomized_range_finder(M, n_random, n_iter)
    # project M to the (k + p) dimensional space
    # using the basis vectors
    B = np.dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B

    U = np.dot(Q, Uhat)
    U, V = svd_flip(U, V)

    return U[:, :n_components], s[:n_components], V[:n_components, :]


def randomized_pca(data, n_componets):
    """
    Randomized PCA based on the scikit-learn
    implementation.

    Parameters
    ----------
    data : 2d array-like

    n_components : int
        Number of PCA components

    Returns
    -------
    (eigenvectors, eigenvalues)
    """
    n_samples, n_features = data.shape

    U, S, V = randomized_svd(data, n_componets)
    eigenvalues = (S ** 2) / n_samples
    eigenvectors = V / S[:, np.newaxis] * np.sqrt(n_samples)

    return eigenvectors, eigenvalues
