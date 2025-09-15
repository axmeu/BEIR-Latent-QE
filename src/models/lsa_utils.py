import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib as plt

def select_k_by_variance(singular_values, tau=0.9, min_k=100, max_k=700):
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    k = np.searchsorted(explained, tau) + 1
    k = min(max(k, min_k), max_k)
    return k, explained


def select_k_by_elbow(tfidf_matrix, max_components=1000):
    n_comp = min(max_components, tfidf_matrix.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=1)
    svd.fit(tfidf_matrix)
    explained_cumsum = np.cumsum(svd.explained_variance_ratio_)

    # finds the elbow by getting the farthest orthogonal point
    x1, y1 = 0, explained_cumsum[0]
    x2, y2 = n_comp - 1, explained_cumsum[-1]

    distances = []
    for i, y in enumerate(explained_cumsum):
        d = np.abs((y2 - y1)*i - (x2 - x1)*(y - y1)) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(d)

    k_elbow = np.argmax(distances) + 1 
    return k_elbow, explained_cumsum


def plot_explained_variance(explained, method_name=""):
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulative")
    title = "Sélection de k"
    if method_name:
        title += f" ({method_name})"
    plt.title(title)
    plt.grid(True)
    plt.show()


def apply_rocchio(query_vec, doc_vectors, alpha=1.0, beta=0.75, gamma=0.25, expansion_nb=30):
    sim_scores = cosine_similarity(query_vec, doc_vectors)[0]

    top_indices_pos = np.argsort(sim_scores)[::-1][:expansion_nb]
    vecs_pos = doc_vectors[top_indices_pos]
    mean_pos = vecs_pos.mean(axis=0).reshape(1, -1)

    bottom_indices_neg = np.argsort(sim_scores)[:expansion_nb]
    vecs_neg = doc_vectors[bottom_indices_neg]
    mean_neg = vecs_neg.mean(axis=0).reshape(1, -1)

    return (alpha * query_vec + beta * mean_pos - gamma * mean_neg)
