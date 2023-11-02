import torch
import torch.nn as nn

class Word2Vec(nn.Module):
  
  def __init__(self, vocabulary_size, embeddings_dimension) -> None:
    super().__init__()
    self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embeddings_dimension, padding_idx=0)
    self.context_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embeddings_dimension, padding_idx=0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, words, positive_contexts, negative_contexts):
    w = words.long().view(-1, 1)
    w_p = self.word_embeddings(w)
    c_p = self.context_embeddings(positive_contexts.long())
    c_n = self.context_embeddings(negative_contexts.long())
    w_c_p = torch.sum(w_p * c_p, dim=2).squeeze()
    w_c_n = torch.sum(w_p * c_n, dim=2).squeeze()
    return self.sigmoid(w_c_p), self.sigmoid(w_c_n)
