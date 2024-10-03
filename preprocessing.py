# Import necessary modules 
!pip install wandb
import wandb

from sklearn.model_selection import train_test_split
import torch


from google.colab import drive
drive.mount('/content/drive')


#Read in the file path
def read_file_path(file_path):
    file_path = file_path 
    # read it in to inspect it
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text



def tokenize_map_ids(example, tokenizer, vocab):
  tokens = tokenizer.encode(example).tokens #convert the strings into tokens
  token_ids = [vocab.get(token) for token in tokens] #convert the tokens into ids
  #token_ids.append(EOS_token) #append the end of sequence token
  return token_ids


def TensorConverter(data):
    data = torch.tensor(data, dtype=torch.long)