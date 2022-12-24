from Bert_Flow_utils import TransformerGlow, AdamWeightDecayOptimizer
from transformers import AutoTokenizer
import random
import pandas as pd

all_tok = pd.read_excel(r'/home/ubuntu/thesis/stance_prediction/Stance_prediction/Wahlprogramme/alle_tokenized.xlsx')

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
# sentences = ['This is sentence A.', 'And this is sentence B.']  # Please replace this with your datasets (single sentences).
sentences = random.sample(all_tok, len(all_tok))
model_inputs = tokenizer(
    sentences,
    add_special_tokens=True,
    return_tensors='pt',
    max_length=659, # longest sentence
    padding='longest',
    truncation=True
)
bertflow.train()
z, loss = bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=True)  # Here z is the sentence embedding
optimizer.zero_grad()
loss.backward()
optimizer.step()

bertflow.save_pretrained('output')  # Save model
bertflow = TransformerGlow.from_pretrained('output')  # Load model

# I've got embeddings for all sentences, how do I get one for a new sentence without retraining the model?