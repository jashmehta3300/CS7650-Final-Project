import torch

def create_data_iter(df, category_dict, tokenizer, target_col):
    """Creates data iterator as list of tuple consisting of `text` and `category`."""
    # Input columns:
    input_cols = set(df.columns) - {target_col}
    
    # Maps category to integer
    df[target_col] = df[target_col].map(category_dict)
    
    # Tokenize samples and create iterator
    iterator = []
    for i in range(len(df)):
        enc_inputs = tokenizer(*([df[c].iloc[i].lower() for c in input_cols]),
                               truncation=True, 
                               padding=False)
        enc_inputs = {k: torch.tensor(v) for k, v in enc_inputs.items()}
        iterator.append({**enc_inputs, "labels": torch.tensor(df[target_col].iloc[i])})
    
    return iterator



class CustomDataset:
    r""" Custom dataset wrapper for hate speech data.
    """

    def __init__(self, data, idxs):
        self.data = data
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.data[self.idxs[item]]
