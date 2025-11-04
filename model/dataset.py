import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    '''
    Arg:
        df: the dataframe with name and sequence infomation of 2.3M representative sequences
        df_name_col: the name of the sequence, which can be used to find the saved embedding
        embed_path: the file path for saved embedding
    '''
    def __init__(self, df, df_name_col, embed_path):
        self.names = df[df_name_col]
        self.embed_path = embed_path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            batch = [self._process_single_item(i) for i in idx]
            return batch
        else:
            return self._process_single_item(idx)

    def _process_single_item(self, idx):
        item = {}

        name = self.names[idx]
        item['name'] = name
        
        file_path = os.path.join(self.embed_path, f'{name}.npz')
        loaded = np.load(file_path)

        item['repr'] = torch.tensor(loaded['repr'][0])

        return item
    
def collate_batch(batch):
    '''
    concatenate tensors for different proteins
    '''
    batch_collated = {}
    batch_collated['repr'] = torch.cat([b['repr'] for b in batch], dim=0)

    return batch_collated