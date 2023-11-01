import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from random import sample
from transformers import BertTokenizer
from model import Word2Vec
import train
from FlattenedDataset import FlattenedDocumentsDataset
import yaml

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

def tokenize_dataset(tokenizer, n_samples=100):
  dataset = load_dataset("scikit-learn/imdb", split="train")
  seed = 42
  shuffled = dataset.shuffle(seed=seed)
  selected = shuffled.select(range(n_samples))
  tokenized = selected.map(lambda x: preprocessing_fn(x, tokenizer), remove_columns=['review', 'sentiment'])
  split = tokenized.train_test_split(test_size=.2, train_size=.8, seed=42)
  return split['train'], split['test']


def collate_fn(batch, radius, scaling_factor, vocab):
  outputs = {
    'word_id':torch.zeros(len(batch)),
    'positive_context_ids':torch.zeros((len(batch), radius*2)),
    'negative_context_ids':torch.zeros((len(batch), radius*scaling_factor))
  }
  for index, (word, context) in enumerate(batch):
    negative_context = sample(vocab, scaling_factor * radius)
    outputs['word_id'][index] = torch.tensor(word)
    outputs['positive_context_ids'][index] = torch.tensor(context)
    outputs['negative_context_ids'][index] = torch.tensor(negative_context)
  return outputs

class Config(dict):
    def __init__(self, config):
        self._conf = config
 
    def __getattr__(self, name):
        if self._conf.get(name) is not None:
            return self._conf[name]
        return None

global DEVICE
DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print("CUDA is available and is used")
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
            "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine.")
    DEVICE = torch.device('cpu')
    print("CUDA and MPS are not available, switching to CPU.")
else:
    DEVICE = torch.device("mps")
    print("CUDA not available, switching to MPS")

def run(config):
  config = Config(config)
  radius = config.radius
  scaling_factor = config.scaling_factor
  n_samples = config.n_samples
  batch_size = config.batch_size
  embedding_dim = config.embedding_dim
  n_epochs = config.n_epochs
  lr = config.lr

  print('START RUN')
  print('-'*100)
  print('Tokenizing Dataset...')
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
  tokenized_train, tokenized_test = tokenize_dataset(tokenizer, n_samples)
  print('Tokenizing Done')
  print('\n')
  print('Extracting token contexts...')
  train_set = FlattenedDocumentsDataset(radius=radius, dataset=tokenized_train['review_ids'])
  test_set = FlattenedDocumentsDataset(radius=radius, dataset=tokenized_test['review_ids'])
  print('Token Contexts Flattened')
  print('\n')
  print('Training...')
  train_dataloader = DataLoader(
      dataset=train_set, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, radius, scaling_factor, list(tokenizer.vocab.values()))
  )
  test_dataloaer = DataLoader(
      dataset=test_set, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, radius, scaling_factor, list(tokenizer.vocab.values()))
  )
  model = Word2Vec(vocabulary_size=len(tokenizer.vocab), embeddings_dimension=embedding_dim)
  model = model.to(DEVICE)
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  train.train_batched(model=model, optimizer=optimizer, num_epochs=n_epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloaer)
  print('Train Done. Saving model.')
  train.save_model(model, optimizer, embedding_dim=embedding_dim, batch=batch_size, epoch=n_epochs, radius=radius, ratio=scaling_factor)


if __name__ == '__main__':
  with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
  run(config=cfg)
      