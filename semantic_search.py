import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

manifesto = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/grüne 2021.xlsx')
print(manifesto)
manifesto_sentences = manifesto['text']

query_embedding = model.encode('öffentlicher Nahverkehr sollte kostenlos sein')
passage_embedding = model.encode(manifesto_sentences)

similarity = util.dot_score(query_embedding, passage_embedding)
value, index = torch.topk(similarity, 10)
with pd.option_context('display.max_colwidth', None):
    print(manifesto_sentences.iloc[index[0]])