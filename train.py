import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from run import DEVICE


def save_model(model, optimizer, embedding_dim, radius, ratio, batch, epoch):
  path = f'model_dim-{embedding_dim}_radius-{radius}_ratio-{ratio}_batch-{batch}_epoch_{epoch}.ckpt'
  torch.save(
    {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, 
    path
  )
  print(f'Model saved to {path}')

def train_batched(model, optimizer, num_epochs, train_dataloader):
  criterion = nn.BCELoss()
  losses = []
  for i in range(num_epochs):
    epoch_loss = 0
    for _, batch in enumerate(tqdm(train_dataloader)):
      optimizer.zero_grad()
      pos_outs, neg_outs = model(batch['word_id'].to(DEVICE), batch['positive_context_ids'].to(DEVICE), batch['negative_context_ids'].to(DEVICE))
      stacked_outs = torch.hstack((pos_outs, neg_outs))
      stacked_reals = torch.hstack((torch.zeros_like(pos_outs), torch.ones_like(neg_outs)))
      loss = criterion(stacked_outs, stacked_reals)
      loss.backward()
      optimizer.step()
      epoch_loss += loss
    losses.append(epoch_loss)
    print('-'*100)
    print(f'Epoch {i} done. Loss: {epoch_loss}')
  
  return losses
