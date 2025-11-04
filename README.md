## MotifAE: Unsupervised Discovery of Functional Motifs from Protein Language Model

MotifAE is a sparse autoencoder with local similarity loss, designed for the unsupervised discovery of functional protein motifs from Protein Language Model. It incorporates a **local similarity loss** to encourage the learned latent features to capture meaningful, coherent local patterns (motifs) within the protein sequences.
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

## Use Pre-trained Models

The weights for the trained **SAE** (Simple Autoencoder) and **MotifAE** models can be downloaded from **Zenodo** (Insert Zenodo link here).

  * **Extracting MotifAE Latent Features:** For a complete example of how to load the model and obtain the MotifAE latent features (motif representations) for a given protein, refer to the provided notebook: `/notebook/example.ipynb`
  * **MotifAE-G for Feature Alignment:** To use **MotifAE-G** for aligning the learned latent features with experimental data (e.g., functional annotations), please refer to the dedicated code and documentation in the `/gate` directory.

-----

## Training the Model. 

To train the MotifAE from scratch, follow the steps below.


### Step 1: Download Representative Protein Data

Download the dataset of **2.3 million representative proteins** from [Zenodo](https://zenodo.org/).  
This dataset was derived through structure-based clustering of the AlphaFold structure database ().

---

### Step 2: Generate ESM-2 Embeddings

Use ESM2-650M to generate last layer embeddings for these representative proteins.

1. Modify Configuration  
   Edit [`model/config.py`](./model/config.py) to specify:
   - The path to your downloaded protein `.csv` files.  
   - The output directory where embeddings will be saved.

2. Run Embedding Script  
   Execute the following command to generate embeddings:

   ```bash
   python model/get_esm2_embedding.py
   ```

---

### Step 3: Train the MotifAE Model

Once the embeddings are ready, you can train the MotifAE model.

1. Modify Configuration
   Edit [`model/config.py`](./model/config.py) again to define the path for saving model checkpoints and log files

1. Run Training Script
   Execute the training script:

   ```bash
   python model/train_esm_sae.py
   ```

