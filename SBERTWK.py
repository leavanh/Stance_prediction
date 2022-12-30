from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import pandas as pd
import logging as lg
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import random

from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead
sys.path.append('/home/ubuntu/thesis/stance_prediction/Stance_prediction')
import SBERTWK_utils



# -----------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# -----------------------------------------------
def trun_pad(sentences):
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params["max_seq_length"]:
            sent_ids = sent_ids[: params["max_seq_length"]]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params["max_seq_length"] - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == params["max_seq_length"]
        assert len(sent_mask) == params["max_seq_length"]

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)
    features_mask = np.array(features_mask)
    return features_input_ids, features_mask

# -----------------------------------------------
# Settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size", default=64, type=int, help="batch size for extracting features."
)
parser.add_argument(
    "--num_epochs", default=2, type=int, help="number of epochs."
)
parser.add_argument(
    "--max_seq_length",
    default=512, #longest sentence in afd is 659, if not 512 error later on
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument(
    "--model_type",
    type=str,
    default="bert-base-german-cased",
    help="Pre-trained language models. (default: 'bert-base-uncased')",
)
parser.add_argument(
    "--embed_method",
    type=str,
    default="ave_last_hidden", #no info in paper, leave default
    help="Choice of method to obtain embeddings (default: 'ave_last_hidden')",
)
parser.add_argument(
    "--context_window_size",
    type=int,
    default=2, #leave default, explained in paper
    help="Topological Embedding Context Window Size (default: 2)",
)
parser.add_argument(
    "--layer_start",
    type=int,
    default=4, #leave default, explained in paper
    help="Starting layer for fusion (default: 4)",
)
args = parser.parse_args()

# -----------------------------------------------
# Set device
torch.cuda.set_device(-1)
device = torch.device("cuda", 0)
args.device = device

# -----------------------------------------------
# Set seed
set_seed(args)
# Set up logger
lg.basicConfig(format="%(asctime)s : %(message)s", level=lg.DEBUG)

# -----------------------------------------------
# Set Model
params = vars(args)

config = AutoConfig.from_pretrained(params["model_type"], cache_dir="./cache")
config.output_hidden_states = True
tokenizer = AutoTokenizer.from_pretrained(params["model_type"], cache_dir="./cache")
model = AutoModelWithLMHead.from_pretrained(
    params["model_type"], config=config, cache_dir="./cache"
)

model.to(params["device"])


# -----------------------------------------------
# Load data
class Manifestos(Dataset):
    def __init__(self):
        # data loading
        self.sentences = list(pd.read_excel(r'/home/ubuntu/thesis/stance_prediction/Stance_prediction/Wahlprogramme/alle_tokenized.xlsx')[0])
        random.shuffle(self.sentences)
        self.n_samples = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return self.n_samples

dataset = Manifestos()
dataloader = DataLoader(dataset=dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

# -----------------------------------------------
# Train model
for epoch in range(params["num_epochs"]):
    embeddings = []
    for i, sentences in enumerate(dataloader):

        features_input_ids, features_mask = trun_pad(sentences)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
        batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        model.zero_grad()

        with torch.no_grad():
            features = model(**inputs)[1]

        # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
        # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
        all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()
        embed_method = SBERTWK_utils.generate_embedding(params["embed_method"], features_mask)
        embedding = embed_method.embed(params, all_layer_embedding)
        embeddings.extend(list(embedding))

# -----------------------------------------------
# Compute similarity
# TODO: not working and also problem how to save model and don't retrain.
query = ["Öffentlicher Nahverkehr soll für alle Personen umsonst sein."]

query_input_ids, query_mask = trun_pad(query)

query_inputs = {"input_ids": torch.tensor(query_input_ids, dtype=torch.long).to(device), "attention_mask": torch.tensor(query_mask, dtype=torch.long).to(device)}
model.zero_grad()

with torch.no_grad():
    features = model(**query_inputs)[1]

query_all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

embed_method = SBERTWK_utils.generate_embedding(params["embed_method"], features_mask)
query_embedding = embed_method.embed(params, query_all_layer_embedding)

len(query_embedding)

