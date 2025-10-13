import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .lsa_utils import (
    apply_rocchio,
    compute_explained_variance,
    select_k_by_elbow,
    plot_explained_variance,
    animate_elbow
)

class LSAModel:
    def __init__(self,
                 # LSA
                 k_components="auto", prefit_components=1000,
                 # Rocchio PRF
                 rocchio_prf=False, window=100, posneg=10, alpha=1.0, beta=0.5, gamma=0.1):

        self.doc_ids = []
        self.doc_texts = []
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32
        )

        # LSA
        self.k_components = k_components          
        self.prefit_components = prefit_components
        self.svd = None
        self.doc_vectors = None

        # Rocchio
        self.rocchio_prf = rocchio_prf
        self.window = window
        self.posneg = posneg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, corpus, plot=False, gif=False):
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [doc["text"] for doc in corpus.values()]
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        # Variance for elbow
        explained_cumsum = compute_explained_variance(
            tfidf_matrix, max_components=self.prefit_components
        )
        if self.k_components == "auto":
            k, _ = select_k_by_elbow(explained_cumsum)
            print(f"k value: {k}")
        else:
            k = int(self.k_components)

        # Project into latent space + normalize
        self.svd = TruncatedSVD(n_components=k, random_state=1, n_iter=10)
        mat_lsa = self.svd.fit_transform(tfidf_matrix)
        self.doc_vectors = normalize(mat_lsa, norm="l2", copy=False).astype(np.float32, copy=False)
        self.svd.components_ = self.svd.components_.astype(np.float32, copy=False)
        del mat_lsa, tfidf_matrix

        # Visuals
        if plot:
            plot_explained_variance(explained_cumsum, k=k, method_name="Elbow")
        if gif:
            animate_elbow(explained_cumsum, k_elbow=k)

    def search(self, query, top_k=10):
        query_tfidf = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_tfidf)
        query_vec = normalize(query_vec, norm="l2", copy=False).astype(np.float32, copy=False)  # (1, k)

        if self.rocchio_prf:
            query_vec = apply_rocchio(
                query_vec, self.doc_vectors, window=self.window,
                posneg=self.posneg, alpha=self.alpha, 
                beta=self.beta, gamma=self.gamma,
            )
            query_vec = query_vec.astype(np.float32, copy=False)
            
        scores = self.doc_vectors @ query_vec.ravel() 
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]
