# conda activate interplm
# nohup python /home/ch3849/SAE_mut/code/model_relu/train_esm_sae.py &

import torch
from torch.utils.data import DataLoader
import pandas as pd
from trainer import StandardTrainer
from training import train_run
from dataset import ProteinDataset, collate_batch
from config import my_config
import esm
import os
import json
import shutil

# Initialize save directory and export config
if my_config['save_dir'] is not None:
    os.makedirs(my_config['save_dir'], exist_ok=True)
    # Copy the code to the save directory
    shutil.copytree('/home/ch3849/SAE_mut/code/model_relu/', my_config['save_dir'] + '/model_relu')

    os.makedirs(os.path.join(my_config['save_dir'], "checkpoints"), exist_ok=True)

    with open(os.path.join(my_config['save_dir'], "config.json"), "w") as f:
        json.dump(my_config, f, indent=4)


stage = my_config['stage']
df = pd.read_csv(my_config[f'df_path_{stage}'])
df = df[df['split'] == 'train'].reset_index(drop=True)

dataset = ProteinDataset(df=df, df_name_col=my_config[f'df_name_col_{stage}'], embed_logit_path=my_config[f'embed_logit_path_{stage}'], stage=stage)
loader = DataLoader(dataset, collate_fn=collate_batch, batch_size=my_config['batch_size'], drop_last=True, num_workers=my_config['dataloader_num_workers'], shuffle = True)

trainer = StandardTrainer(my_config)
train_run(data=loader, trainer=trainer, my_config=my_config)