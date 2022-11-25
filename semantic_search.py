import pandas as pd
import numpy as np
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
nltk.download('punkt')

# functions
def get_sent_list(manifesto: pd.Series) -> list:
    """
    The party manifesto gets split into sentences
    """
    if not isinstance(manifesto, pd.Series):
        raise TypeError("manifesto has to be a pd.Series")
    tokenized = list()
    for paragraph in manifesto:
        sentences = nltk.sent_tokenize(paragraph)
        tokenized.append(sentences)

    flattened = [item for sublist in tokenized for item in sublist]
    return flattened

# read all manifestos and split into sentences
grüne = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/grüne 2021.xlsx')['text']
grüne_tok = get_sent_list(grüne)

query_embedding = model.encode('öffentlicher Nahverkehr sollte kostenlos sein')
passage_embedding = model.encode(grüne_tok)

similarity = util.dot_score(query_embedding, passage_embedding)
value, index = torch.topk(similarity, 5)
with pd.option_context('display.max_colwidth', None):
    print(manifesto_sentences.iloc[index[0]])