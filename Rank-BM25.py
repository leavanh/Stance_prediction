from rank_bm25 import BM25Okapi
import pandas as pd

sentences = list(pd.read_excel(r'Wahlprogramme/alle_tokenized.xlsx')[0])

tokenized_corpus = [s.split(" ") for s in sentences]

bm25 = BM25Okapi(tokenized_corpus)

query = "Öffentlicher Nahverkehr sollte gratis für alle sein"
tokenized_query = query.split(" ")

bm25.get_top_n(tokenized_query, sentences, n=1)