import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .lsa_utils import apply_rocchio, init_knn, knn_search

class LSAModel:
    def __init__(self,
                 # LSA
                 k_components="auto", tau=0.9, min_k=100, max_k=700, prefit_components=1000,
                 # Rocchio PRF
                 rocchio_prf=False, expansion_nb=30, alpha=1.0, beta=0.75, gamma=0.25,
                 # kNN latent
                 knn_search=False, k_knn=5):
        
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

        # kNN
        self.knn_search = knn_search
        self.k_knn = k_knn
        self.nn_model = None

    def select_k_by_variance(self, S):
        explained = np.cumsum(S**2) / np.sum(S**2)
        k = np.searchsorted(explained, self.tau) + 1
        k = min(max(k, self.min_k), self.max_k)
        return k

    def fit(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [doc["text"] for doc in corpus.values()]
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        if self.k_components == "auto":
            prefit = TruncatedSVD(
                n_components=min(self.prefit_components, tfidf_matrix.shape[1]-1),
                random_state=1
            )
            prefit.fit(tfidf_matrix)
            k = self.select_k_by_variance(prefit.singular_values_)
            print(f"[INFO] Composantes retenues : {k} (≥ {self.tau*100:.0f}% variance)")
            self.svd = TruncatedSVD(n_components=k, random_state=1)
        else:
            self.svd = TruncatedSVD(n_components=self.k_components, random_state=1)

        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

        if self.knn_search:
            self.nn_model = init_knn(self.doc_vectors, self.k_knn)

    def search(self, query, top_k=10):
        query_tfidf = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_tfidf)

        if self.rocchio_prf:
            query_vec = apply_rocchio(query_vec, self.doc_vectors,
                                      alpha=self.alpha, beta=self.beta,
                                      gamma=self.gamma, expansion_nb=self.expansion_nb)

        if self.knn_search:
            return knn_search(query_vec, self.nn_model, self.doc_ids, top_k)

        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]

