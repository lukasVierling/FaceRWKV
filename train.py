import os, sys, time, math, random, json, datetime, logging
import numpy as np
import torch
from torch.utils.data import Dataset
from src.trainer import Trainer, TrainerConfig
from src.utils import set_seed

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

# RWKV       : our new model - fastest when ctx_len is long - VRAM friendly - good performance
# MHA_rotary : usual MultiheadAttention+Rotary+GeGLU - not as good
# MHA_shift  : with time-shift - good performance
# MHA_pro    : slow (lots of tricks) - VRAM hungry - very good performance

datafile = #path to image data set
datafile_encoding = 'utf-8' #NOT NEEDED

datafile_type = 0 # use 0 for char-level english. use 1 for chinese. only affects some RWKV hyperparametrs NOT NEEDED

#################################### VERY IMPORTANT ####################################
epoch_save_frequency = 10                            # 0 = never, 1 = every 'epoch', 2 = every two 'epoch', etc.
epoch_save_path = 'trained-'

batch_size = 32                                      # if you see "CUDA out of memory", reduce this.
                                                     # if you have good GPU, increase this.
                                                     # use GPU-Z to find the highest value for your VRAM.

n_epoch = 100                                        # the 'epoch' here is actually very short (and of fixed length)
########################################################################################

model_level = 'character' # 'character' (recommended) or 'word' NONEED

ctx_len = 256 # context length, try 512 or 1024 if you have good GPU
n_layer = 6   # try 12 for 100M, 24 for 300M
n_head = 8    # try 12 for 100M, 16 for 300M

n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

lr_init = 6e-4 #if model_type == 'RWKV' else 4e-4    # RWKV can use higher lr.  8e-4 = 0.0008   4e-4 = 0.0004
lr_final = 4e-5

betas = (0.9, 0.99) #if model_type == 'RWKV' else (0.9, 0.99)
eps = 4e-9
weight_decay = 0 #if model_type == 'RWKV' else 0.01  # wd is not useful when we have enough data

epoch_length_fixed = 10000                          # make an 'epoch' very short, so we can see the training progress

######## special hyperparameters for RWKV model ########
rwkv_emb_scale = 0.4                                # scale of initial embedding. 0.4 is a good choice
rwkv_tiny_attn = 0#64 if (datafile_type == 0 and ctx_len > 600) else 0 # extra tiny attention dim, useful for long ctx char-level english
rwkv_tiny_head = 1                                  # 1 is good enough. 8 is slow
# n_side_proj = 512                                 # extra 'side projection', quite useful for BPE models 

# Start training
config = TrainerConfig(
    datafile=datafile,
    datafile_encoding=datafile_encoding,
    datafile_type=datafile_type,
    model_type=model_type,
    model_level=model_level,
    ctx_len=ctx_len,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    n_attn=n_attn,
    n_ffn=n_ffn,
    lr_init=lr_init,
    lr_final=lr_final,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
    epoch_length_fixed=epoch_length_fixed,
    rwkv_emb_scale=rwkv_emb_scale,
    rwkv_tiny_attn=rwkv_tiny_attn,
    rwkv_tiny_head=rwkv_tiny_head,
    # n_side_proj=n_side_proj,
)
# Get dataset
dataset = Dataset(config)
# Get dataloader fro training, validation, and testing
train_loader, valid_loader, test_loader = dataset.get_dataloader(batch_size)
# Get model
model = model(config)

# Start train loop
for epoch in range(n_epoch):
    # Train
    for batch in train_loader:
        # Get batch data
        batch = batch.to(config.device)
        # Train model

        # Print loss
        print(f'Epoch {epoch} | Loss {loss:.4f}')