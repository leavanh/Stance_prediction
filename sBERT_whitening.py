import torch
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from semantic_search import get_sent_list #ToDo: fix, not working yet

query = "Elternunabhängiges Bafög für alle Studierenden"

grüne = pd.read_excel(r'stance_prediction/Stance_prediction/Wahlprogramme/grüne 2021.xlsx')['text']
grüne_tok = get_sent_list(grüne)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

query_embedding = model.encode(query)
grüne_embedding = model.encode(grüne_tok)
grüne_embedding.shape

def whitening_torch(embeddings): #check if does the same as with tensors, check dims
    mu = np.mean(embeddings, axis=0, keepdims=True)
    cov = np.matmul(np.transpose((embeddings - mu)), embeddings - mu)
    u, s, vh = np.linalg.svd(cov)
    W = np.matmul(u, np.diag(1/np.sqrt(s)))
    embeddings = np.matmul(embeddings - mu, W)
    return embeddings, mu, W

grüne_white, mu, W = whitening_torch(grüne_embedding)
query_white = np.matmul(query_embedding - mu, W)

similarity = util.dot_score(query_white, grüne_white)
value, index = torch.topk(similarity, 5)
grüne_sim = [grüne_tok[i] for i in index.tolist()[0]]