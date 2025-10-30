my_config = {
    "stage":'representative', # 'human' or 'representative'

    "layer": 33,
    "plm_name": 'esm2_t33_650M_UR50D',
    "embed_logit_path_representative": '/nfs/scratch/ch3849/esm_sae/esm_output/esm2_650M_embed_logit',
    "embed_logit_path_human": '/nfs/scratch/ch3849/esm_sae/human_proteome/esm2_650M_embed_LLR/embed',

    # "plm_name": 'esm1b_t33_650M_UR50S',
    # "embed_logit_path_representative": '/nfs/user/Users/ch3849/esm_sae/esm_output/esm1b_embed_logit',
    # "embed_logit_path_human": '/nfs/user/Users/ch3849/esm_sae/human_proteome/esm1b_embed_LLR/embed',    
    
    # "df_path": '/share/vault/Users/ch3849/esm_sae/sequence/eval_test_seq_max1022_addmask.csv',
    "df_path_representative": '/nfs/user/Users/ch3849/esm_sae/sequence/training_seq_max1022_addmask.csv',
    "df_name_col_representative": 'uniprot', 
    "df_seq_col_representative": 'random_subregion',

    "df_path_human": '/nfs/user/Users/ch3849/esm_sae/human_proteome/embed_name_unique_sequence.csv',
    "df_name_col_human": 'name', 

    "save_dir": '/nfs/user/Users/ch3849/esm_sae/model/250417',
    
    "device": "cuda:2",
    "batch_size": 40,
    "batch_token": None,
    "noise_sd": None,
    "dataloader_num_workers": 10,
    "activation_dim": 1280,
    "dict_size": int(1280*32),
    "resample_steps": None, # resample the weight for the feature every resample_steps
    "resample_training_steps": 0, # only resample the weight in the first resample_steps_training_percent*total_steps
    "tied": False,
    
    "lr": 1e-3,
    "warmup_steps": 500, # learning rate warmup

    "l1_penalty": 0.04,
    "smooth_penalty": 1,
    "l1_annealing_steps": 5000,
    "seed": 42,

    "n_epoch_representative": 2,
    "n_epoch_human": 10,
    "start_step_representative": 0,
    "start_step_human": 110_000,
    "save_steps_representative": 5_000,
    "save_steps_human": 5_000,
    "log_steps": 20,
}