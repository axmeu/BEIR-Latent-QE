import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def apply_rocchio(query_vec, doc_vectors, alpha=1.0, beta=0.75, gamma=0.25, expansion_nb=30):
    sim_scores = cosine_similarity(query_vec, doc_vectors)[0]

    top_indices_pos = np.argsort(sim_scores)[::-1][:expansion_nb]
    vecs_pos = doc_vectors[top_indices_pos]
    mean_pos = vecs_pos.mean(axis=0).reshape(1, -1)

    bottom_indices_neg = np.argsort(sim_scores)[:expansion_nb]
    vecs_neg = doc_vectors[bottom_indices_neg]
    mean_neg = vecs_neg.mean(axis=0).reshape(1, -1)

    return (alpha * query_vec + beta * mean_pos - gamma * mean_neg)


def init_knn(doc_vectors, k_knn=10):
    nn_model = NearestNeighbors(n_neighbors=k_knn, metric="cosine")
    nn_model.fit(doc_vectors)
    return nn_model


def knn_search(query_vec, nn_model, doc_ids, top_k=10):
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_k)
    return [(doc_ids[i], 1 - distances[0][idx]) for idx, i in enumerate(indices[0])]
