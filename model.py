import torch
import torch.nn as nn

class Word2Vec(nn.Module):
  
  def __init__(self, vocabulary_size, embeddings_dimension) -> None:
    super().__init__()
    self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embeddings_dimension, padding_idx=0)
    self.context_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embeddings_dimension, padding_idx=0)
    self.sigmoid = nn.Sigmoid()
    
  def forward_2(self, word, context):
    w = self.word_embeddings(word.long())
    c = self.context_embeddings(context.long())
    self.sigmoid(torch.dot(w, c))

  def forward(self, word, positive_contexts, negative_contexts):
    w = self.word_embeddings(word.long())
    w_p = w.repeat((len(positive_contexts), 1))
    w_n = w.repeat((len(negative_contexts), 1))
    c_p = self.context_embeddings(positive_contexts.long())
    c_n = self.context_embeddings(negative_contexts.long())
    w_c_p = torch.einsum ('ij, ij -> i', w_p, c_p)
    w_c_n = torch.einsum ('ij, ij -> i', w_n, c_n)
    return self.sigmoid(w_c_p), self.sigmoid(w_c_n)

