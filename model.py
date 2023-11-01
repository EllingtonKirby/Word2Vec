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

  def forward(self, words, positive_contexts, negative_contexts):
    w = words.long().view(-1, 1)
    w_p = w.repeat(1, positive_contexts.shape[1])
    w_n = w.repeat(1, negative_contexts.shape[1])
    w_p = self.word_embeddings(w_p)
    w_n = self.word_embeddings(w_n)
    c_p = self.context_embeddings(positive_contexts.long())
    c_n = self.context_embeddings(negative_contexts.long())
    w_c_p = torch.einsum ('ijk, ijk -> ij', w_p, c_p).flatten()
    w_c_n = torch.einsum ('ijk, ijk -> ij', w_n, c_n).flatten()
    return self.sigmoid(w_c_p), self.sigmoid(w_c_n)

