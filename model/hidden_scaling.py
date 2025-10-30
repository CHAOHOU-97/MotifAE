# conda activate interplm
# nohup python /home/ch3849/SAE_mut/code/model_relu/hidden_scaling.py &

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from trainer import StandardTrainer
from training import train_run
from dataset import ProteinDataset, collate_batch
from config import my_config
from dictionary import AutoEncoder


####################################
# Load the normalization matrix and dataset
####################################
if my_config["plm_name"] == 'esm2_t33_650M_UR50D':
    norm = torch.load('/share/vault/Users/ch3849/esm_sae/model/normalize_vector/ESM2_2k.pth')
elif my_config["plm_name"] == 'esm1b_t33_650M_UR50S':
    norm = torch.load('/share/vault/Users/ch3849/esm_sae/model/normalize_vector/ESM1b_2k.pth')
norm_mean = norm['mean'].requires_grad_(False).to(my_config['device'])
norm_std = norm['std'].requires_grad_(False).to(my_config['device'])

stage = my_config['stage']
df = pd.read_csv(my_config[f'df_path_{stage}'])
df = df[df['split'] == 'train'].reset_index(drop=True)

dataset = ProteinDataset(df=df, df_name_col=my_config[f'df_name_col_{stage}'], embed_logit_path=my_config[f'embed_logit_path_{stage}'], stage=stage)
loader = DataLoader(dataset, collate_fn=collate_batch, batch_size=my_config['batch_size'], drop_last=True, num_workers=my_config['dataloader_num_workers'], shuffle = True)

####################################
# Load the SAE model and checkpoint
####################################
sae_model = 250219
chk = 110000
chk_path = f'/share/vault/Users/ch3849/esm_sae/model/{sae_model}/checkpoints/step_{chk}.pt'
sae = AutoEncoder.from_pretrained(chk_path)
sae.eval()  # disables dropout for deterministic results
sae = sae.to(my_config['device'])

####################################
# Train the scaling factor
####################################
f_scaling = torch.nn.Parameter(torch.ones(my_config['dict_size'], device=my_config['device']))  # scaling factor for each feature
optimizer = optim.Adam([f_scaling], lr=0.1)

for batch_data in loader:
    act = batch_data['repr']
    act = (act.to(my_config["device"]) - norm_mean) / norm_std  # Normalize the input

    f = sae.encode(act)  # Encode the input
    act_hat = sae.decode(f * f_scaling)  # Apply scaling and decode
    l2_loss = torch.linalg.norm(act - act_hat, dim=-1).mean()
    
    optimizer.zero_grad()
    l2_loss.backward()
    optimizer.step()
    f_scaling.data = torch.clamp(f_scaling.data, min=1)  # Ensure scaling factor is positive

    # Optional: print the loss for monitoring
    print(f"L2 Loss: {l2_loss.item()}")