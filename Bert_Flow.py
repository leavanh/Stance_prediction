import sys
from transformers import AutoTokenizer
import random
import pandas as pd
from sentence_transformers import util
import torch
import pickle

sys.path.append('/home/ubuntu/lrz/thesis/Stance_prediction')
from Bert_Flow_utils import TransformerGlow, AdamWeightDecayOptimizer

random.seed(910)

all_tok = list(pd.read_excel(r"/home/ubuntu/lrz/thesis/Stance_prediction/Wahlprogramme/alle_tokenized.xlsx")[0])

model_name_or_path = 'bert-base-german-cased'
bertflow = TransformerGlow(model_name_or_path, pooling='first-last-avg')  # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters= [
    {
        "params": [p for n, p in bertflow.glow.named_parameters()  \
                        if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in bertflow.glow.named_parameters()  \
                        if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamWeightDecayOptimizer(
    params=optimizer_grouped_parameters, 
    lr=1e-3, 
    eps=1e-6,
)

# Important: Remember to shuffle your training data!!! This makes a huge difference!!!
sentences = random.sample(all_tok, len(all_tok))
model_inputs = tokenizer(
    sentences,
    add_special_tokens=True,
    return_tensors='pt',
    max_length=512, # longest sentence
    padding='max_length',
    truncation=True
)
bertflow.train()
z, loss = bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=True)  # Here z is the sentence embedding
optimizer.zero_grad()
loss.backward()
optimizer.step()

bertflow.save_pretrained('/home/ubuntu/lrz/thesis/Stance_prediction/Models/bert-flow-model')  # Save model

with open("/home/ubuntu/lrz/thesis/Stance_prediction/Embeddings/all_bertflow_embeddings", "wb") as fp:   # Save the embeddings
    pickle.dump(z, fp)

with open("/home/ubuntu/lrz/thesis/Stance_prediction/Embeddings/all_bertflow_sentences", "wb") as fp:   # Save the sentences
    pickle.dump(sentences, fp)


bertflow = TransformerGlow.from_pretrained('/home/ubuntu/lrz/thesis/Stance_prediction/Models/bert-flow-model')  # Load model

with open("/home/ubuntu/lrz/thesis/Stance_prediction/Embeddings/all_bertflow_embeddings", "rb") as fp:   # Open the saved embeddings
    embeddings = pickle.load(fp)

with open("/home/ubuntu/lrz/thesis/Stance_prediction/Embeddings/all_bertflow_sentences", "rb") as fp:   # Open the saved embeddings
    sentences = pickle.load(fp)

query = ["Studierende sollen elternunabhägiges Bafög bekommen."]
query = ['Das über einen Zeitraum von 30 Jahren angekündigte Programm zum Wiederaufbau und zur Entwicklung ist keine Entschädigungsleistung, sondern muss als technische Entwicklungshilfe bewertet werden.']
model_input = tokenizer(
    query,
    add_special_tokens=True,
    return_tensors='pt',
    max_length=512, # so it's the same length as the other sentences
    padding='max_length', # so it's the same length as the other sentences
    truncation=True
)

query_embedding = bertflow(model_input['input_ids'], model_input['attention_mask'])

# get similarity
similarity = util.dot_score(query_embedding, embeddings)
value, index = torch.topk(similarity, 5)
[all_tok[i] for i in index.tolist()[0]]