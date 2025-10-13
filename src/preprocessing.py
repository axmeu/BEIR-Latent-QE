import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

def tokenize_doc(doc):
    return [
        (t.lemma_ or t.text).lower()
        for t in doc
        if t.is_alpha and not t.is_space and not t.is_punct
           and not t.is_stop
           and len(t.text) >= 2
    ]

def clean_data(corpus: dict, queries: dict):
    # Corpus
    print("Preprocessing corpus...")
    ids = list(corpus.keys())
    texts = [(d.get("title","") + " " + d.get("text","")) for d in corpus.values()]
    cleaned_corpus = {}
    for i, doc in zip(ids, nlp.pipe(texts, batch_size=256)):
        cleaned_corpus[i] = {"text": " ".join(tokenize_doc(doc))}

    # Queries
    print("Preprocessing queries...")
    qids = list(queries.keys())
    qtexts = list(queries.values())
    cleaned_queries = {}
    for qid, doc in zip(qids, nlp.pipe(qtexts, batch_size=256)):
        cleaned_queries[qid] = {"text": " ".join(tokenize_doc(doc))}
    return cleaned_corpus, cleaned_queries
