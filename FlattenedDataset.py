from torch.utils.data import Dataset

class FlattenedDocumentsDataset(Dataset):
  def __init__(self, word_contexts=None, radius=None, dataset=None) -> None:
    if radius != None and dataset != None:
      self.word_contexts = self.flatten_dataset_to_list(dataset, radius)
    elif word_contexts != None:
      self.word_contexts = word_contexts

  def __getitem__(self, index):
    return (self.word_contexts[0][index], self.word_contexts[1][index])
  
  def __len__(self):
    return len(self.word_contexts[0])

  def flatten_dataset_to_list(self, dataset, radius):
    words = []
    contexts = []
    for document in dataset:
      doc_w, doc_c = self.extract_words_contexts(document, radius)
      words += doc_w
      contexts += doc_c
    return words, contexts
  
  def extract_words_contexts(self, document: list, radius):
    words = []
    contexts = []
    for (index, word) in enumerate(document):
      words.append(word)
      context = []
      start = index-radius
      end = index+radius+1
      for context_index in range(start, end):
        if context_index == index:
          continue
        elif context_index < 0 or context_index >= len(document):
          context.append(0)
        else: 
          context.append(document[context_index])
      contexts.append(context)
    return words, contexts
