import pandas as pd
import numpy as np
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
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

def sBERT(query: str, manifesto: list, topk: int = 5) -> list:
    """
    return the top k similar sentences using a pretrained sBERT model
    """
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    query_embedding = model.encode(query)
    sent_embedding = model.encode(manifesto)

    similarity = util.dot_score(query_embedding, sent_embedding)
    value, index = torch.topk(similarity, 5)
    return [manifesto[i] for i in index.tolist()[0]]


# read all manifestos and split into sentences
grüne = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/grüne 2021.xlsx')['text']
grüne_tok = get_sent_list(grüne)
fdp = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/fdp 2021.xlsx')['text']
fdp_tok = get_sent_list(fdp)
linke = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/linke 2021.xlsx')['text']
linke_tok = get_sent_list(linke)
spd = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/spd 2021.xlsx')['text']
spd_tok = get_sent_list(spd)
cdu = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/cdu 2021.xlsx')['text']
cdu_tok = get_sent_list(cdu)
afd = pd.read_excel(r'/Users/lea_vanheyden/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Masterarbeit/Polit Language Model/Data/Wahlprogramme/afd 2021.xlsx')['text']
afd_tok = get_sent_list(afd)

# get similarity from sBERT
query = "Nahverkehr soll für alle Personen umsonst sein."

grüne_sim = sBERT(query, grüne_tok)
with pd.option_context('display.max_colwidth', None):
    print(grüne_sim)

fdp_sim = sBERT(query, fdp_tok)
with pd.option_context('display.max_colwidth', None):
    print(fdp_sim)

linke_sim = sBERT(query, linke_tok)
with pd.option_context('display.max_colwidth', None):
    print(linke_sim)
    
spd_sim = sBERT(query, spd_tok)
with pd.option_context('display.max_colwidth', None):
    print(spd_sim)
    
cdu_sim = sBERT(query, cdu_tok)
with pd.option_context('display.max_colwidth', None):
    print(cdu_sim)

afd_sim = sBERT(query, afd_tok)
with pd.option_context('display.max_colwidth', None):
    print(afd_sim)

