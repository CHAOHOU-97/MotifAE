# conda activate interplm
# nohup python ~/SAE_mut/code/InterPLM/interplm/train/perplexity.py &

import sys, os
import pandas as pd
import numpy as np

import esm
from scipy.special import softmax

# Load tokenizer
esm1b, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

def calculate_perplexity(logit, input):
    '''
    logit: the esm2 output logit, dim: L*33
    input: the token of unmasked sequence and masked sequence

    return: mean perplexity of 15% masked residues
    '''
    ppl = -np.log(softmax(logit, axis=1))
    msk_ppl = ppl[input[1] == alphabet.mask_idx]
    msk_residue = input[0][input[1] == alphabet.mask_idx]
    
    return float(np.mean([msk_ppl[i, msk_residue[i]] for i in range(len(msk_residue))]))


df = pd.read_csv('/share/vault/Users/ch3849/esm_sae/sequence/eval_test_seq_max1022_addmask_perplexity.csv')

for i in df.index:
    name = df.loc[i, 'uniprot']
    data = [(name, df.loc[i, seq]) for seq in ['random_subregion', 'masked_sequence']]

    labels, strs, input = batch_converter(data)
    input = input.numpy()[:,1:-1]

    try:
        loaded = np.load(f'/share/vault/Users/ch3849/esm_sae/esm1b_embed_eval_withmask/{name[-2:]}/{name}.npz')
        logit = loaded['logit']
        df.loc[i, 'esm1b_perplexity'] = calculate_perplexity(logit, input)
    except:
        print(f'{i} error')

df.to_csv('/share/vault/Users/ch3849/esm_sae/sequence/eval_test_seq_max1022_addmask_perplexity.csv', index=False)
