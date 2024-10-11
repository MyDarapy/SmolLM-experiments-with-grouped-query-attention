import torch 

#Read in the file path
def read_file_path(file_path):
    file_path = file_path 
    # read it in to inspect it
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def TensorConverter(data):
    data = torch.tensor(data, dtype=torch.long)