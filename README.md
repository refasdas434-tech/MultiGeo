
## 🛠 Installation

 This project requires a specific combination of PyTorch 2.1.0 and CUDA 12.1 to ensure compatibility with DGL and avoid dependency conflicts. Please follow the steps below strictly.

### 1. Create and Activate Environment
```bash
conda create -n multigeo python=3.9 -y
conda activate multigeo
```

### 2. Install PyTorch 2.1.0 (CUDA 12.1)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Graph Neural Network Libraries
Install DGL and PyG compatible with PyTorch 2.1:

```bash
# Install DGL
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Install TorchData (Required by DGL)
pip install torchdata==0.7.0

# Install PyG (PyTorch Geometric) and dependencies
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### 4. Apply Critical Patches
Fix NumPy version conflicts and missing NVIDIA libraries:

```bash
pip install "numpy<2.0"

# Install missing NVIDIA shared libraries manually
pip install nvidia-cusparse-cu12 nvidia-cublas-cu12
```

### 5. Configure Environment Variables (Crucial)
DGL requires explicit paths to the CUDA libraries installed in step 4. Run the following command in your terminal:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```

> **Note:** The `export` command above is only valid for the current session. For permanent configuration, please add it to your `~/.bashrc`.

### 6. Verify Installation
Run the following command to check if DGL loads correctly:

```bash
python -c "import dgl; import torch; print(f'DGL Backend: {dgl.backend.backend_name}'); print('Success')"
```

## 🚀 Reproduction

We provide pre-trained models and specific configuration files to reproduce the Cold-Start experimental results on the Davis and KIBA dataset.

### 1. File Locations
- **Configuration Files**: Located in `checkpoints/davis_best_three/assets/configs/`. This directory contains the configuration files for the three cold-start settings.
- **Pre-trained Weights**: Located in `checkpoints/MultiGeo_davis_dyn_disagreement/`.

### Running Inference

Use the `inference.py` script to evaluate the model. You can reproduce the results for **Cold Drug**, **Cold Target**, or **Cold Both** settings by specifying the `--split_type` and pointing to the corresponding checkpoint.

#### Example: Cold Drug Setting
```bash
python inference.py \
  --data_root data/Davis \
  --protein_conformer_dir data/Davis/Davis_results \
  --model_path checkpoints/MultiGeo_davis_dyn_disagreement/Davis/cold_drug/best_model.pt \
  --dataset Davis \
  --split_type cold_drug
```


## 📂 Data Availability

Due to the significant file size of the datasets and pre-processed features, they cannot be hosted directly on this GitHub repository. Furthermore, to strictly adhere to the double-blind review policy and maintain anonymity during the submission process, we are temporarily withholding external download links.

We will release the full dataset and pre-trained weights via a public cloud storage link  immediately upon the paper's acceptance/publication.
