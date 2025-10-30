my_config = {
    # before training the model, please set the following paths
    "embed_path": 'your_embed_path',
    "df_path": 'your_dataframe_path.csv', # download it from zenodo
    "save_dir": 'your_save_dir',

    "df_name_col": 'uniprot', 
    "df_seq_col": 'random_subregion',

    "device": "cuda:0",
    "batch_size": 40,
    "dataloader_num_workers": 10,
    "activation_dim": 1280,
    "dict_size": int(1280*32),
    "resample_steps": None, # resample the weight for the feature every resample_steps
    "tied": False,
    
    "lr": 1e-3,
    "warmup_steps": 500, # learning rate warmup

    "l1_penalty": 0.04, # for SAE, set l1_penalty to 0.85
    "local_similarity_penalty": 1, # this is the relative value compared to L1 penalty. for SAE, set local_similarity_penalty to 0
    "l1_annealing_steps": 5000,
    "seed": 42,

    "n_epoch": 2,
    "save_steps": 5_000,
    "log_steps": 20,
}