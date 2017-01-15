import scipy
import numpy as np
import scipy.sparse as sp


def pagerank(graph_matrix, n_iter=100, alpha=0.9, tol=1e-6):
    n_nodes = graph_matrix.shape[0]
    n_edges_per_node = graph_matrix.sum(axis=1)
    n_edges_per_node = np.array(n_edges_per_node).flatten()

    np.seterr(divide='ignore')
    normilize_vector = np.where((n_edges_per_node != 0),
                                1. / n_edges_per_node, 0)
    np.seterr(divide='warn')

    normilize_matrix = sp.spdiags(normilize_vector, 0,
                                  *graph_matrix.shape, format='csr')

    graph_proba_matrix = normilize_matrix * graph_matrix

    teleport_proba = np.repeat(1. / n_nodes, n_nodes)
    is_dangling, = scipy.where(normilize_vector == 0)

    x_current = teleport_proba
    for _ in range(n_iter):
        x_previous = x_current.copy()

        dangling_total_proba = sum(x_current[is_dangling])
        x_current = (
            x_current * graph_proba_matrix +
            dangling_total_proba * teleport_proba
        )
        x_current = alpha * x_current + (1 - alpha) * teleport_proba

        error = np.abs(x_current - x_previous).mean()
        if error < tol:
            break

    else:
        print("PageRank didn't converge")

    return x_current
