## MotifAE: Unsupervised Discovery of Functional Motifs from Protein Language Model

MotifAE is a sparse autoencoder with an additional smoothness loss, designed for the unsupervised discovery of functional sequence patterns from Protein Language Model (we used ESM2-650M here). The smoothness loss encourages the latent features to capture meaningful, coherent patterns in the protein sequences.
![MotifAE architecture](./image/motifae.png)

-----

## Prerequisites
For MotifAE training and running, please set up the environment first:

```bash
conda create -n motifae python=3.12
conda activate motifae
pip install fair-esm pandas numpy
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```
Other environments with the necessary dependencies should also work as long as PyTorch and ESM are properly installed.

-----

## Use Pre-trained Models

The weights for the trained MotifAE model can be downloaded from [Zenodo](https://zenodo.org/records/17488191).

### Compare MotifAE Features with Known Motifs
Please refer to 
[`notebook/ELM_motif.ipynb`](./notebook/ELM_motif.ipynb) for instructions on how to:
- Load a trained MotifAE model,  
- Extract latent features from protein sequences,   
- Compare these latent features with known annotations (e.g., ELM motifs),


### MotifAE-G for Feature Alignment with Experimental Data and Protein Design
To use MotifAE-G for aligning latent features with experimental data — for example, domain folding stability — follow the steps below:

1. The information for proteins with stability mutational scanning is provided in [`data/412pro_info.csv`](./data/412pro_info.csv). Please download the corresponding mutation effect data (412pros_ddG_ML.csv) from [Zenodo](https://zenodo.org/records/17488191).
2. Process these data using code in [`gate/1_stability_prepare.ipynb`](./gate/1_stability_prepare.ipynb)
3. Train MotifAE-G to align latent features with experimental measurements: [`gate/2_gate_model.ipynb`](./gate/2_gate_model.ipynb)  
4. Designing proteins with enhanced stability by steering stability-associated latent features: [`gate/3_protein_design.ipynb`](./gate/3_protein_design.ipynb)  

-----

## Training the Model. 

To train the MotifAE from scratch, follow the steps below.

### Step 1: Download Representative Protein Sequences

Download the dataset of 2.3 million representative proteins from [Zenodo](https://zenodo.org/records/17488191).  
This dataset was derived through [structure-based clustering of the AlphaFold structure database](https://afdb-cluster.steineggerlab.workers.dev/).


### Step 2: Generate ESM-2 Embeddings

Generate ESM2-650M last layer embeddings for these representative proteins.

1. Edit [`model/config.py`](./model/config.py) to specify:
   - The path to your downloaded protein `representative_2.3M_seq.csv` files.  
   - The output directory where embeddings will be saved.

2. Execute the following command to generate embeddings (this takes a few days to finish on one GPU):

   ```bash
   python model/get_esm2_embedding.py
   ```

### Step 3: Train the MotifAE Model

Once the embeddings are ready, you can train the MotifAE model.

1. Edit [`model/config.py`](./model/config.py) to define the path for saving model checkpoints and log files

1. Run Training Script (finish in one day on one GPU)
   ```bash
   python model/train_esm_sae.py
   ```

-----
## Citation
MotifAE Reveals Functional Sequence Patterns from Protein Language Model: Unsupervised Discovery and Interpretability Analysis. Chao Hou, Di Liu, Yufeng Shen. BioRxiv, November 2025. [link](https://www.biorxiv.org/content/10.1101/2025.11.04.686576v2)