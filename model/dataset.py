import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re

class ProteinDataset(Dataset):
    '''
    return the sequence token, ESM embedding and final logits
    Arg:
        df: the dataframe with name and sequence infomation of 2.3M representative sequences
        df_name_col: the name of the sequence, which can be used to find the saved embedding and logits
        embed_logit_path: the file path for saved embedding and logits 
    '''
    def __init__(self, df, df_name_col, embed_logit_path, stage):
        self.names = df[df_name_col]
        self.embed_logit_path = embed_logit_path
        self.stage = stage

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
        
        file_path = os.path.join(self.embed_logit_path, re.split(r'[-_]', name)[0][-2:], f'{name}.npz') # name.split('-')[0] can get the name for isoform
        loaded = np.load(file_path)

        if self.stage == 'representative':
            item['repr'] = torch.tensor(loaded['repr'])
            # item['logit'] = torch.tensor(loaded['logit'])
        elif self.stage == 'human':
            item['repr'] = torch.tensor(loaded['arr_0'])

        return item
    
def collate_batch(batch):
    '''
    concatenate tensors for different proteins
    '''
    batch_collated = {}
    batch_collated['repr'] = torch.cat([b['repr'] for b in batch], dim=0)

    # batch_collated['logit'] = torch.cat([b['logit'] for b in batch], dim=0)

    return batch_collated