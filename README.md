# BEIR-Latent-methods-explorations
This project explores **classical Information Retrieval (IR)** using the [BEIR benchmark](https://github.com/beir-cellar/beir).  
I implemented **BM25 and Latent Semantic Analysis (LSA)** and experimented with a **Rocchio-like PRF** (Pseudo Relevance Feedback), built on a **zero-shot LSA with heuristically selected k components**. 

---

## Objectives
- Reproduce **classical baselines** (BM25, LSA).  
- Study the impact of **dimensionality reduction (LSA)** with heuristacally k component selected.
- Explore if **query expansion (Rocchio-like feedback)** can improve retrieval in semantic space.
- Evaluate methods on **NFCorpus, SciDocs, ArguAna, Scifact**  BEIR datasets.  

---

## Methods
- **BM25**: A strong bag-of-words ranking function.  
- **LSA (Truncated SVD)**: Project TF-IDF into a latent semantic space by reducing dimensions.
- **Zero-shot LSA**: k selected component with a **cumulative explained variance method** and elbow method, on a lower component prefit matrix.
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



**Takeaway:**  
Naïve Rocchio in latent space is **unstable** → sometimes small improvements, often degradations.  
This reflects a common limitation of pseudo-relevance feedback: it can reinforce noise and is sensitive to parameter tuning.  

---

## Key Takeaways
- **BM25 remains the strongest baseline**, confirming its robustness.
- **LSA** depends on the type of document and on the size of vocabulary.
- **Rocchio feedback did not consistently improve results**   

---

## Repository structure

