from tokenizers import ByteLevelBPETokenizer
import json

# Initialize a tokenizer
def tokenizer_trainer(data_files, saving_directory, vocab_size):
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer on your dataset
    tokenizer.train(files = data_files, vocab_size=vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    ])

    # Save the trained tokenizer
    tokenizer.save_model(saving_directory)
    print('tokenizer successfully trained')

    return tokenizer


#Get the vocabulary created by the tokenizer
def get_vocab(vocab_file_path):
    vocab_file_path = vocab_file_path
    with open(vocab_file_path, 'r') as file:
        vocab = json.load(file)
    return vocab 

