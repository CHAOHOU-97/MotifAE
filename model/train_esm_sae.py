from torch.utils.data import DataLoader
import pandas as pd
from trainer import StandardTrainer
from training import train_run
from dataset import ProteinDataset, collate_batch
from config import my_config
import os
import json

# Initialize save directory and export config
if my_config['save_dir'] is not None:
    os.makedirs(my_config['save_dir'], exist_ok=True)

    os.makedirs(os.path.join(my_config['save_dir'], "checkpoints"), exist_ok=True)

    with open(os.path.join(my_config['save_dir'], "config.json"), "w") as f:
        json.dump(my_config, f, indent=4)

# read data
df = pd.read_csv(my_config['df_path'], nrows=10)
# df = df[df['split'] == 'train'].reset_index(drop=True) # we used 2.2M randomly selected sequences for training

# create dataloader
dataset = ProteinDataset(df=df, df_name_col=my_config['df_name_col'], embed_path=my_config['embed_path'])
loader = DataLoader(dataset, collate_fn=collate_batch, batch_size=my_config['batch_size'], drop_last=True, num_workers=my_config['dataloader_num_workers'], shuffle = True)

# train the model
trainer = StandardTrainer(my_config)
train_run(data=loader, trainer=trainer, my_config=my_config)