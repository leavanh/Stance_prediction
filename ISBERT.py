"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
import sys
sys.path.append('/home/ubuntu/lrz/thesis/IS-BERT')
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = '/home/ubuntu/lrz/thesis/IS-BERT/examples/training/nli/datasets/'
sts_dataset_path = '/home/ubuntu/lrz/thesis/IS-BERT/examples/training/nli/datasets/stsbenchmark.tsv.gz'


if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-german-cased'

# Read the dataset
train_batch_size = 32


model_save_path = '/home/ubuntu/lrz/thesis/IS-BERT/examples/training/nli/output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

cnn = models.CNN(in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(cnn.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, cnn, pooling_model])


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

train_samples = []

sentences = gzip.open(os.path.join(nli_dataset_path, 'nli_sentences.train.gz'),
               mode="rt", encoding="utf-8").readlines()

all_tok = list(pd.read_excel(r'/home/ubuntu/lrz/thesis/Stance_prediction/Wahlprogramme/alle_tokenized.xlsx')[0])
sentences = all_tok

for s in sentences:
    sentence = s.strip().split('\t')[0]
    label_id = 1
    train_samples.append(InputExample(texts=[sentence], label=1))


train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MutualInformationLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension())


#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)

embeddings = model.encode(sentences, convert_to_numpy = True)
len(sentences)
embeddings.shape
sentence = sentences[:-1]
embedding = model.encode(sentence, convert_to_numpy = True)
embedding == embeddings[:-1] # very weird behaviour, some embeddings are the same and some are not
