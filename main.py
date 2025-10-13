#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from src.preprocessing import clean_data
from src.data_loader import data_loader
from src.evaluation import evaluate_model
from src.models.bm25 import BM25Model
from src.models.lsa import LSAModel

def run_bm25(corpus, queries, qrels):
    bm25 = BM25Model()
    bm25.fit(corpus)
    print("bm25 loaded")
    ndcg, _map, recall, precision = evaluate_model(bm25, queries, qrels)
    print(f'bm25:\nNDCG@10: {ndcg["NDCG@10"]:.3f}, Recall@100: {recall["Recall@100"]:.3f}')

def run_lsa(corpus, queries, qrels, k="auto", rocchio=False, plot=False, gif=False, prefit_components=1000):
    k_components = "auto" if str(k).lower() == "auto" else int(k)
    lsa = LSAModel(
        k_components=k_components,
        prefit_components=prefit_components,
        rocchio_prf=rocchio
    )
    lsa.fit(corpus, plot=plot, gif=gif)
    label = "lsa_prf" if rocchio else "lsa"
    print(f"{label} loaded")
    ndcg, _map, recall, precision = evaluate_model(lsa, queries, qrels)
    print(f'{label}:\nNDCG@10: {ndcg["NDCG@10"]:.3f}, Recall@100: {recall["Recall@100"]:.3f}')

def run_all(corpus, queries, qrels):
    bm25 = BM25Model()
    bm25.fit(corpus)
    print("bm25 loaded")
    lsa = LSAModel()
    lsa.fit(corpus)
    print("lsa loaded")
    lsa_prf = LSAModel(rocchio_prf=True)
    lsa_prf.fit(corpus)
    print("lsa + rocchio loaded")
    models = {"bm25": bm25, "lsa": lsa, "lsa_prf": lsa_prf}
    for name, model in models.items():
        ndcg, _map, recall, precision = evaluate_model(model, queries, qrels)
        print(f'{name}:\nNDCG@10: {ndcg["NDCG@10"]:.3f}, Recall@100: {recall["Recall@100"]:.3f}')

def main():
    parser = argparse.ArgumentParser(description="Experiment on a zero-shot LSA (heuristic-k) on BEIR datasets with BM25 baseline")
    parser.add_argument("--dataset", type=str, required=True, choices=["arguana", "scifact", "nfcorpus", "scidocs"])
    parser.add_argument("--model", type=str, required=True, choices=["bm25", "lsa", "all"])
    parser.add_argument("--k", default="auto")
    parser.add_argument("--rocchio", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--prefit-components", type=int, default=1000)
    args = parser.parse_args()

    corpus, queries, qrels = data_loader(args.dataset)
    cleaned_corpus, cleaned_queries = clean_data(corpus, queries)

    if args.model == "bm25":
        run_bm25(cleaned_corpus, cleaned_queries, qrels)
    elif args.model == "lsa":
        run_lsa(cleaned_corpus, cleaned_queries, qrels,
                k=args.k, rocchio=args.rocchio,
                plot=args.plot, gif=args.gif,
                prefit_components=args.prefit_components)
    else:
        run_all(cleaned_corpus, cleaned_queries, qrels)

if __name__ == "__main__":
    main()


#python .\main.py --dataset nfcorpus --model bm25
