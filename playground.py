import torch 
import torch.nn as nn
import wandb 

from preprocessing import read_file_path, tokenize_map_ids, TensorConverter 
from transformer import SmolLM
from train_tokenizer import tokenizer, get_vocab

def split_data(data, train_data_size):
  n = int(0.8*len(data))
  train_dataset = data[:n]
  test_dataset = data[n:]
  return train_dataset, test_dataset


class DataLoader:
    def__init__(self, data, context_length, batch_size):
        self.data = data
        self.batch_size = batch_size 
        self.context_length = context_length
        self.n_data = len(data)
        self.no_of_batches = int(self.n_data/self.batch_size)

    
    def__len__(self):
        return self.batch_size



    def load_dataset(self, split):
        for _ in range(self.no_of_batches):
            self.data = train_data if split == 'train' else test_data
            idx = torch.randint(len(self.data) - self.context_length, (self.batch_size,)) #idx has size of {batch_size}
            x = torch.LongTensor([self.data[i:i+self.context_length] for i in idx])
            y = torch.LongTensor([self.data[i+1:i+self.context_length+1] for i in idx]) #The target this includes the next token we are trying to predict

            x,y = x.to(device), y.to(device)
            yeild x, y



# Model evaluation code
@torch.no_grad()
def calculate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = load_dataset(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


# Function to save a checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
data = 
vocab_file_path = 
data_files = 
saving_filepath = 
context_length
batch_size = 
vocab_size = 
T_max = 
  def main():
    dataset = read_file_path(data)

    train_data, test_data = split_data(dataset, train_data_size=0.8)
    
    tokenizer = tokenizer(data_files, saving_filepath, vocab_size, min_frequency)

    vocab = get_vocab(vocab_file_path)

    train_data = tokenize_map_ids(text, tokenizer, vocab)
    test_data = tokenize_map_ids(text, tokenizer, vocab)


    train_dataloader = DataLoader(train_data, context_length, batch_size)
    test_dataloader = DataLoader(test_data, context_length, batch_size)

    train_iter = iter(train_dataset.load_dataset(split='train'))


    model =SmolLM().to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    wandb.init(project='smol LLM Design Experiments')

    for iter in range(max_iter):
        if iter % evaluation_intervals == 0 or iter == max_iter - 1:
            losses = calculate_loss()
            checkpoint = {
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss}
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{iter}.pth.tar")
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "learning_rate": scheduler.get_last_lr()[0]})

        # Sample a batch of data
        for x, y in enumerate(train_dataloader)
  
        #Evaluate the loss, calculate gradient, update weight
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == "__main__":

    



    

