from rank_bm25 import BM25Okapi
import numpy as np

class BM25Model:
    def __init__(self, k1=0.9, b=0.4):
        self.doc_ids = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.k1 = k1
        self.b = b

    def fit(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.tokenized_corpus = [
            corpus[doc_id]["text"].split() for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query, top_k=10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_indices]