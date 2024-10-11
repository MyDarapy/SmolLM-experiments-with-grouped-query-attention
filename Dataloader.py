import torch


def split_data(data, train_data_size):
  n = int(0.8*len(data))
  train_dataset = data[:n]
  test_dataset = data[n:]
  return train_dataset, test_dataset

def tokenize_map_ids(example, tokenizer, vocab):
  tokens = tokenizer.encode(example).tokens #convert the strings into tokens
  token_ids = [vocab.get(token) for token in tokens]#convert the tokens into ids
  #token_ids.append(EOS_token) #append the end of sequence token
  return token_ids

class DataLoader:
    def __init__(self, data, block_size, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_data = len(data)
        self.no_of_batches = int(self.n_data/self.batch_size)


    def __len__(self):
        return self.no_of_batches

    def load_dataset(self):
        for _ in range(self.no_of_batches):
          idx = torch.randint(len(self.data) - self.block_size, (self.batch_size,)) #idx has size of {batch_size}
          x = torch.LongTensor([self.data[i:i+self.block_size] for i in idx])
          y = torch.LongTensor([self.data[i+1:i+self.block_size+1] for i in idx]) #The target this includes the next token we are trying to predict

          x,y = x.to(device), y.to(device)
          yield x, y
