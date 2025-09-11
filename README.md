# BEIR-Latent-Methods-Rocchio-and-kNN-for-Query-Expansion
This project explores **classical Information Retrieval (IR)** using the [BEIR benchmark](https://github.com/beir-cellar/beir).  
I implemented **BM25 and Latent Semantic Analysis (LSA)** and experimented with a **Rocchio-like PRF** (Pseudo Relevance Feedback) and **kNN query expansion in the latent space**, built on a **zero-shot LSA with heuristically selected k components**. 

---

## Objectives
- Reproduce **classical baselines** (BM25, LSA).  
- Study the impact of **dimensionality reduction (LSA)**.  
- Explore whether **query expansion (Rocchio-like feedback)** can improve retrieval in semantic space.  
- Evaluate methods on **NFCorpus** and **SciDocs** datasets.  

---

## Methods
- **TF-IDF**: Sparse lexical baseline.  
- **BM25**: A strong bag-of-words ranking function.  
- **LSA (Truncated SVD)**: Project TF-IDF into a latent semantic space (dim = 200–1200).  
- **Rocchio-like feedback**: Modify query vectors by shifting towards top-ranked documents (positive feedback) and away from bottom ones (negative feedback).  

Metrics used:
- **nDCG@10** → ranking quality  
- **Recall@100** → coverage of relevant documents  

---

## Results

### nDCG@10
| Dataset           |  TF-IDF  |  BM25  |  LSA   | LSA + PRF |
|-------------------|----------|--------|--------|-----------|
| NFCorpus          |   0.304  |  0.314 |  **0.328** |   0.322   |
| Scidocs           |   **0.150**  |  0.146 |  0.126 |   0.116   |


### Recall@100
| Dataset           |  TF-IDF  |  BM25  |  LSA   | LSA + PRF |
|-------------------|----------|--------|--------|-----------|
| NFCorpus          |   0.241  |  0.239 |  0.309 |   **0.325**   |
| Scidocs           |   0.354  |  0.343 |  0.351 |   **0.358**   |

---

## Analysis of Rocchio Feedback
I tested different values of **β (positive weight)** and **γ (negative weight)** for the Rocchio-like update.  

- The settings we used did not show gains in NDCG@10. Our PRF with Rocchio in latent space systematically did worse than LSA.
- However, PRF brought a better recall@100 **marginal gains** on all the corpus we tested (~ +0.02 on NFCorpus for example).   
- Example of expansion parameters tested on NFCorpus:

| expansion size | β   | γ   | nDCG@10 | Recall@100 |
|----------------|-----|-----|----------|-------------|
| / | 0 | 0.0 | |      |
| 10 | 0.75 | 0.25 |    |       |
| 30 | 0.6 | 0.4 |  |       |
| 50 | 0.0 | 0.0 |   |        |

**Takeaway:**  
Naïve Rocchio in latent space is **unstable** → sometimes small improvements, often degradations.  
This reflects a common limitation of pseudo-relevance feedback: it can reinforce noise and is sensitive to parameter tuning.  

---

## Key Takeaways
- **BM25 remains the strongest baseline**, confirming its robustness.  
- **LSA benefits from higher dimensionality** (~700 for NFCorpus, ~1200 for SciDocs).  
- **Rocchio feedback did not consistently improve results**, but analyzing these failures was informative:
  - Showed the risks of negative feedback in latent spaces.  
  - Highlighted why more advanced feedback methods (kNN-based, embedding-based PRF) are used in modern IR.  

---

## Repository structure

