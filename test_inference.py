import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tabulate import tabulate
from datasets import load_dataset
from random import sample
from tqdm import tqdm
from transformers import BertTokenizer
from train import DEVICE


class Advanced_Conv1dClassifier(nn.Module):
    """A text classifier:
    - input = minibatch
    - output = probability associated to a binary classification task
    - vocab_size: the number of words in the vocabulary we want to embed
    - embedding_dim: size of the word vectors
    """

    def __init__(self, vocab_size, embedding_dim, embeddings=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        if embeddings != None:
          self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
          self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)

        self.conv_1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=100, kernel_size=3, padding=1)
        self.pooling_1 = nn.AdaptiveMaxPool1d(output_size=253)
        self.flatten_1 = nn.Flatten()

        self.conv_2 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=100, kernel_size=5, padding=2)
        self.pooling_2 = nn.AdaptiveMaxPool1d(output_size=251)
        self.flatten_2 = nn.Flatten()
        
        self.conv_3 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=100, kernel_size=7, padding=3)
        self.pooling_3 = nn.AdaptiveMaxPool1d(output_size=249)
        self.flatten_3 = nn.Flatten()

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=.5)
        self.linear_layer = nn.LazyLinear(out_features=2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids):
        embedding_output = self.embedding(input_ids)
        output_1 = self.flatten_1(self.pooling_1(self.relu(self.conv_1(embedding_output.mT))))
        output_2 = self.flatten_2(self.pooling_2(self.relu(self.conv_2(embedding_output.mT))))
        output_3 = self.flatten_3(self.pooling_3(self.relu(self.conv_3(embedding_output.mT))))
        output = self.dropout(torch.concatenate((output_1, output_2, output_3), dim=-1))
        linear_output = self.linear_layer(output)
        return self.sigmoid(linear_output)

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # `batch` is a list of dictionary with keys "review_ids" and "label".
        features = [{"input_ids": x["review_ids"]} for x in batch]
        features = self.tokenizer.pad(
            features, padding="max_length", max_length=256, return_tensors="pt"
        )
        label = torch.tensor([x["label"] for x in batch])[:, None]
        return {"review_ids": features["input_ids"], "label": label}

def preprocessing_fn(x, tokenizer):
    x["review_ids"] = tokenizer(
        x["review"],
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    x["label"] = 0 if x["sentiment"] == "negative" else 1
    return x

def prep_dataset(tokenizer):
  dataset = load_dataset("scikit-learn/imdb", split="train")
  n_samples = 5000  # the number of training example
  seed = 42
  dataset = dataset.shuffle(seed=seed)
  dataset = dataset.select(range(n_samples))
  dataset = dataset.map(lambda x: preprocessing_fn(x, tokenizer), batched=False)
  dataset = dataset.remove_columns(['review','sentiment'])
  dataset = dataset.train_test_split(train_size=.8, seed=42)
  train_set = dataset['train']
  valid_set = dataset['test']
  return train_set, valid_set

def load_model(path='./model_dim-4_radius-2_ratio-3_batch-4_10.ckpt'):
  model_state = torch.load(path)['model_state_dict']
  word_embeddings = model_state['word_embeddings.weight']
  context_embeddings = model_state['context_embeddings.weight']
  return word_embeddings, context_embeddings

def train(path=None, embeddings_dim=4, batch_size=32, lr=1e-2):
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
  print('Loading Dataset...')
  train_set, valid_set = prep_dataset(tokenizer)
  data_collator = DataCollator(tokenizer)
  train_dataloader = DataLoader(
      train_set, batch_size=batch_size, collate_fn=data_collator
  )
  valid_dataloader = DataLoader(
      valid_set, batch_size=batch_size, collate_fn=data_collator
  )
  
  if path != None:
    word_embeddings, context_embeddings = load_model(path=path)
  else:
     word_embeddings = None

  loss_function = nn.BCELoss()

  model = Advanced_Conv1dClassifier(
      vocab_size=len(tokenizer.vocab), embedding_dim=embeddings_dim, embeddings=word_embeddings
  )
  model = model.to(DEVICE)
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  print('Training...')
  n_epochs = 10 
  for e in range(n_epochs):
      train_loss_per_epoch = []
      train_accuracy = 0
      for i, batch in enumerate(tqdm(train_dataloader)):
          optimizer.zero_grad()
          input, target = batch['review_ids'].to(DEVICE), batch['label'].to(DEVICE)
          output = model(input)
          loss = loss_function(output, F.one_hot(target).reshape(-1, 2).float())
          loss.backward()
          optimizer.step()
          train_loss_per_epoch.append(loss.item())
          train_accuracy += torch.sum(torch.argmax(output, dim=1) == target.squeeze())
      
      # Validation
      valid_loss_per_epoch = []
      valid_accuracy = 0
      with torch.no_grad():
          for i, batch in enumerate(tqdm(valid_dataloader)):
              input, target = batch['review_ids'].to(DEVICE), batch['label'].to(DEVICE)
              output = model(input)
              loss = loss_function(output, F.one_hot(target).reshape(-1, 2).float())
              valid_loss_per_epoch.append(loss.item())
              valid_accuracy += torch.sum(torch.argmax(output, dim=1) == target.squeeze())

      print('-'*100)
      print(f'Epoch: {e}')
      print(f'Train Loss: {np.array(train_loss_per_epoch).mean()}')
      print(f'Valid Loss: {np.array(valid_loss_per_epoch).mean()}')
      print(f'Train Accuracy: {train_accuracy / len(train_dataloader.dataset)}')
      print(f'Valid Accuracy: {valid_accuracy / len(valid_dataloader.dataset)}')
  return model
    

if __name__ == '__main__':
   train(
      path=None,
      embeddings_dim=100,
      batch_size=32,
      lr=1e-2,
   )
