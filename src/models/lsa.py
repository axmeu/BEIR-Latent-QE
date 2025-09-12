import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LSAModel:
    def __init__(self, n_components="auto", use_query_expansion=True, expansion_nb=30,
                 alpha=1.0, beta=0.75, gamma=0.25, tau=0.9, min_k=100, max_k=700,
                 prefit_components=1000):

        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.n_components = n_components
        self.tau = tau
        self.min_k = min_k
        self.max_k = max_k
        self.prefit_components = prefit_components
        self.svd = None
        self.doc_vectors = None
        self.doc_ids = []
        self.doc_texts = []
        self.use_query_expansion = use_query_expansion
        self.expansion_nb = expansion_nb
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def select_k_by_variance(self, S):
        explained = np.cumsum(S**2) / np.sum(S**2)
        k = np.searchsorted(explained, self.tau) + 1
        k = min(max(k, self.min_k), self.max_k)
        return k

    def fit(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [doc["text"] for doc in corpus.values()]
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        if self.n_components == "auto":
            prefit = TruncatedSVD(
                n_components=min(self.prefit_components, tfidf_matrix.shape[1]-1),
                random_state=1
            )
            prefit.fit(tfidf_matrix)
            k = self.select_k_by_variance(prefit.singular_values_)
            print(f"[INFO] Nombre de composantes retenues : {k} (variance cumulative ≥ {self.tau*100:.0f}%)")
            self.svd = TruncatedSVD(n_components=k, random_state=1)
        else:
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=1)

        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

    def search(self, query, top_k=10):
        query_tfidf = self.vectorizer.transform([query])
        query_lsa = self.svd.transform(query_tfidf)

        if self.use_query_expansion:
            sim_scores_init = cosine_similarity(query_lsa, self.doc_vectors)[0]

            top_indices_pos = np.argsort(sim_scores_init)[::-1][:self.expansion_nb]
            vecs_pos = self.doc_vectors[top_indices_pos]
            mean_pos = vecs_pos.mean(axis=0).reshape(1, -1)

            bottom_indices_neg = np.argsort(sim_scores_init)[:self.expansion_nb]
            vecs_neg = self.doc_vectors[bottom_indices_neg]
            mean_neg = vecs_neg.mean(axis=0).reshape(1, -1)

            query_lsa = (self.alpha * query_lsa +
                         self.beta * mean_pos -
                         self.gamma * mean_neg)

        scores = cosine_similarity(query_lsa, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]

