import re
import torch
import esm
import os, pickle
import pandas as pd
import numpy as np
from config import my_config
import random

############################
# load data and model
############################
df = pd.read_csv(my_config['df_path'])
max_length = 1022

# Function to sample a random subregion of 1022 residues
def random_sample(sequence, max_length):
    if len(sequence) > max_length:
        start_idx = random.randint(0, len(sequence) - max_length)
        sequence = sequence[start_idx:start_idx + max_length]
    return sequence
df[my_config['df_seq_col']] = df['sequence'].apply(random_sample, max_length = max_length)

# Load ESM model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

############################
# get and save embeddings
############################
for i in df.index:
    data = [(df.loc[i, my_config['df_name_col']], df.loc[i, my_config['df_seq_col']])]

    batch_labels, batch_strs, batch_input = batch_converter(data)
    batch_input = batch_input.to(device)

    with torch.no_grad():
        outputs = model(batch_input, repr_layers=[33])
        
    repr = outputs['representations'][33].cpu()

    label = batch_labels[0]
    # organize by the last two letters of the uniprot id in folders
    save_path = os.path.join(my_config['embed_logit_path'], re.split(r'[-_]', label)[0][-2:], f'{label}.npz')

    np.savez_compressed(save_path, repr=repr.numpy())