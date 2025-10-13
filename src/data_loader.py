from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

# scifact
# arguana
# nfcorpus
# scidocs

def data_loader(dataset: str):
    print("Loading dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "data")

    corpus_path = os.path.join(data_path, "corpus.jsonl")
    queries_path = os.path.join(data_path, "queries.jsonl")
    qrels_path = os.path.join(data_path, "qrels", "test.tsv")

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path,
        query_file=queries_path,
        qrels_file=qrels_path
    ).load_custom()
    return corpus, queries, qrels

