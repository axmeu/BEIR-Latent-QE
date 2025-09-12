import spacy

def tokenize(s):
    nlp = spacy.load("en_core_web_sm")
    s = nlp(s)
    tokens = [token.text.lower()
              for token in s
              if not token.is_punct and not token.is_stop and not token.is_space]
    return tokens

def clean_data(corpus: dict, queries: dict):
    cleaned_corpus = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        full_text = title + " " + text
        tokens = tokenize(full_text)
        cleaned_corpus[doc_id] = {"text": " ".join(tokens)}

    cleaned_queries = {}
    for query_id, query_text in queries.items():
        tokens = tokenize(query_text)
        cleaned_queries[query_id] = {"text": " ".join(tokens)}

    return cleaned_corpus, cleaned_queries