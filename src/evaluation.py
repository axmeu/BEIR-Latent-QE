from beir.retrieval.evaluation import EvaluateRetrieval

def evaluate_model(model, queries, qrels, k_values=[10, 100]):
    print("Computing results...")
    results = {}
    max_k = max(k_values)

    for qid in qrels:
        if qid not in queries:
            continue
        query_text = queries[qid]["text"]
        scores = model.search(query_text, top_k=max_k)
        results[qid] = {doc_id: float(score) for doc_id, score in scores}

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=k_values)
    return (ndcg, _map, recall, precision)

