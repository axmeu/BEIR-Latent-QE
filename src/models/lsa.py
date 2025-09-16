import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .lsa_utils import (
    apply_rocchio,
    compute_explained_variance,
    select_k_by_variance,
    select_k_by_elbow,
    plot_explained_variance,
    animate_elbow
)

class LSAModel:
    def __init__(self,
                 # LSA
                 k_components="auto", tau=0.9, min_k=100, max_k=700, prefit_components=1000,
                 # Rocchio PRF
                 rocchio_prf=False, expansion_nb=30, alpha=1.0, beta=0.75, gamma=0.25):

        self.doc_ids = []
        self.doc_texts = []

        # LSA
        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.k_components = k_components
        self.tau = tau
        self.min_k = min_k
        self.max_k = max_k
        self.prefit_components = prefit_components
        self.svd = None
        self.doc_vectors = None

        # Rocchio
        self.rocchio_prf = rocchio_prf
        self.expansion_nb = expansion_nb
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, corpus, variance_cum=False, elbow=False, plot=False, tau=None, gif=False):
        if tau is not None:
            self.tau = tau

        self.doc_ids = list(corpus.keys())
        self.doc_texts = [doc["text"] for doc in corpus.values()]
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        explained_cumsum = compute_explained_variance(tfidf_matrix, max_components=self.prefit_components)

        # Select k
        if self.k_components == "auto":
            if variance_cum:
                k, _ = select_k_by_variance(explained_cumsum, tau=self.tau,
                                            min_k=self.min_k, max_k=self.max_k)
                print(f"[INFO] k retenu par variance cumulative : {k} (≥ {self.tau*100:.0f}% variance)")

            elif elbow:
                k, _ = select_k_by_elbow(explained_cumsum)
                print(f"[INFO] k retenu par méthode elbow : {k}")
            else:
                k = self.min_k
        else:
            k = self.k_components

        self.svd = TruncatedSVD(n_components=k, random_state=1)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

        # Plot
        if plot:
            plot_explained_variance(explained_cumsum, k=k, method_name="Variance cumulée")

        # GIF
        if gif and elbow:
            animate_elbow(explained_cumsum, k_elbow=k)


    def search(self, query, top_k=10):
        query_tfidf = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_tfidf)

        if self.rocchio_prf:
            query_vec = apply_rocchio(query_vec, self.doc_vectors,
                                      alpha=self.alpha, beta=self.beta,
                                      gamma=self.gamma, expansion_nb=self.expansion_nb)

        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]
