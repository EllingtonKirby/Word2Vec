import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

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

def train_batched(model, optimizer, num_epochs, train_dataloader, valid_dataloader):
  criterion = nn.BCELoss()
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
  losses = []
  for i in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    print('-'*100)
    print('Training:')
    for _, batch in enumerate(tqdm(train_dataloader)):
      optimizer.zero_grad()
      pos_outs, neg_outs = model(batch['word_id'].to(DEVICE), batch['positive_context_ids'].to(DEVICE), batch['negative_context_ids'].to(DEVICE))
      stacked_outs = torch.hstack((pos_outs, neg_outs))
      stacked_reals = torch.hstack((torch.ones_like(pos_outs), torch.zeros_like(neg_outs)))
      loss = criterion(stacked_outs, stacked_reals)
      loss.backward()
      optimizer.step()
      epoch_loss += loss
      epoch_accuracy += torch.sum(torch.round(stacked_outs) == stacked_reals) / torch.numel(stacked_reals)
    losses.append(epoch_loss)

    print('-'*100)
    print('Validating:')
    with torch.no_grad():
      valid_loss = 0
      valid_accuracy = 0
      for _, batch in enumerate(tqdm(valid_dataloader)):
        pos_outs, neg_outs = model(batch['word_id'].to(DEVICE), batch['positive_context_ids'].to(DEVICE), batch['negative_context_ids'].to(DEVICE))
        stacked_outs = torch.hstack((pos_outs, neg_outs))
        stacked_reals = torch.hstack((torch.ones_like(pos_outs), torch.zeros_like(neg_outs)))
        loss = criterion(stacked_outs, stacked_reals)
        valid_loss += loss
        valid_accuracy += torch.sum(torch.round(stacked_outs) == stacked_reals) / torch.numel(stacked_reals)
    print('-'*100)
    print(f'Epoch {i} done.')
    print(f'Train Loss: {epoch_loss / len(train_dataloader)}')
    print(f'Validaton loss: {valid_loss / len(valid_dataloader)}')
    print(f'Train Accuracy: {epoch_accuracy / len(train_dataloader)}')
    print(f'Validation Accuracy: {valid_accuracy / len(valid_dataloader)}')
    scheduler.step(valid_loss / len(valid_dataloader))

def test_model(model, dataloader):
    criterion = nn.BCELoss()
    print('-'*100)
    print('Testing:')
    with torch.no_grad():
      valid_loss = 0
      valid_accuracy = 0
      for _, batch in enumerate(tqdm(dataloader)):
        pos_outs, neg_outs = model(batch['word_id'].to(DEVICE), batch['positive_context_ids'].to(DEVICE), batch['negative_context_ids'].to(DEVICE))
        stacked_outs = torch.hstack((pos_outs, neg_outs))
        stacked_reals = torch.hstack((torch.ones_like(pos_outs), torch.zeros_like(neg_outs)))
        loss = criterion(stacked_outs, stacked_reals)
        valid_loss += loss
        valid_accuracy += torch.sum(torch.round(stacked_outs) == stacked_reals) / torch.numel(stacked_reals)
    print(f'Test loss: {valid_loss / len(dataloader)}')
    print(f'Test Accuracy: {valid_accuracy / len(dataloader)}')
