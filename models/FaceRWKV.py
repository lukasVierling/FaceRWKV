import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import yaml
import torchvision.models as models
from models.Preprocessing import LinearSequencing, CNNSequencing

from models.RWKV import Block

class FaceRWKV(nn.Module):
    def __init__(self, config):
        """
        A Face Recognition model utilizing RWKV blocks.

        Args:
        - config: Configuration for the model.

        """
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dim = config.n_embd
        self.n_layers = config.n_layer
        self.n_classes = config.n_classes
        self.mean = config.mean
        self.pos_enc = config.pos_enc
        self.n_ffn = config.n_ffn
        self.mlp_head = config.mlp_head
        self.block = config.block
        self.resnet = config.resnet

        if self.resnet:
            self.sequencing = CNNSequencing(self.patch_size, self.embed_dim)
            self.num_patches = 20*30 #TODO : check if this holds XD
            config.ctx_len = self.num_patches #reset the context length
        else:
            self.sequencing = LinearSequencing(self.patch_size, self.embed_dim)
            self.num_patches = config.resolution[0]*config.resolution[1]//(config.patch_size**2)
            config.ctx_len = self.num_patches #reset the context length

        # Learned positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        # RWKV Blocks
        if self.block == "rwkv":
            self.blocks = nn.Sequential(*[Block(config, i) for i in range(self.n_layers)])
        elif self.block == "transformer":
            #use transformer blocks
            self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.n_head, dim_feedforward=self.n_ffn, batch_first=True), num_layers=self.n_layers)
        elif self.block == "identity":
            # flatten the resnet output for the mlp mlp_head
            self.blocks = nn.Flatten()

        # MLP mlp_Head
        if self.mlp_head:
            # if we operate directly on the resnet feature map, we have to use a different input size for the mlp
            if self.block == "identity":
                embed_dim_1 = 20*30*self.embed_dim #TODO Change this for 
            else:
                embed_dim_1 = self.embed_dim

            self.head = nn.Sequential(
                nn.Linear(embed_dim_1, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.n_classes)
            )
        else:
            self.head = nn.Linear(self.embed_dim, self.n_classes)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input images, shape (batch_size, 3, H, W).

        Returns:
        - Predictions for classes, shape (batch_size, n_classes).

        """
        # x.shape = (batch_size, 3, H, W)
        # Reshape to patches
        x = self.sequencing(x)
        if self.pos_enc:
            x = x + self.pos_embedding    
        # x.shape = (batch_size, 14*14, embed_dim)
        #print if grad is calc
        x = self.blocks(x)
        # Extract last hidden state
        # x.shape = (batch_size, n_patches, embed_dim) -> (batch_size, embed_dim)
        #x = x[:, -1, :]
        if self.block == "identity":
            # flatten the resnet output for the mlp mlp_head
            x = x.unsqueeze(1)
        if self.mean:
            x = torch.mean(x, dim=1)
        else: 
            x = x[:, -1, :]
        # x.shape = (batch_size, n_classes)
        x = self.head(x)
        return x

class RWKVConfig:
    def __init__(self):
        # Model architecture parameters
        self.n_embd = 256           # Embedding size
        self.n_attn = 4             # Number of attention mlp_heads
        self.n_head = 4             # Number of mlp_heads for RWKV_TinyAttn
        self.ctx_len = 256          # Context length -> apparently crashes for too short context????
        #self.vocab_size = 50000    # Vocabulary size
        self.rwkv_emb_scale = 1.0   # Scale for final projection in RWKV_TimeMix and RWKV_ChannelMix
        self.rwkv_tiny_attn = 64    # Tiny attention size for RWKV_TinyAttn
        self.rwkv_tiny_head = 2     # Number of tiny attention mlp_heads for RWKV_TinyAttn
        self.n_ffn = 512            # Hidden size for RWKV_ChannelMix
        self.n_layer = 4            # Number of RWKV blocks
        self.patch_size = 20        # Size of patches to be extracted from input images
        self.n_classes = 7          # Number of output classes
        self.resolution = (600,400)
        self.mean = False           # will calculate mean over sequence if true, else take last hidden state (perfroms better when turned off)
        # Initialization parameters
        self.scale_init = 0         # Scale for weight initialization in RWKV_TimeMix and RWKV_ChannelMix
        self.pos_enc = True         # Whether to use positional encoding    
        self.rwkv = True            # When true use rwkv blocks, else use transformer blocks
        self.block = "rwkv"         # Which block to use, options are "rwkv", "transformer", "identity"
        self.mlp_head = False        # Whether to use an mlp mlp_head or a linear mlp_head
        self.resnet = False

    def calculate_decay_speed(self, h):
        return math.pow(self.ctx_len, -(h + 1) / (self.n_head - 1))
    
    def from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        model_config = config_dict.get('model', {})
        for k, v in model_config.items():
            setattr(self, k, v)

