########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, config): # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters(): # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0 # extra scale for gain

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: # final projection?
                    scale = config.rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: # token emb?
                    scale = config.rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) # the distance
            for h in range(config.n_head):
                if h < config.n_head - 1:
                    decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        # if config.rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(config.n_attn, config.n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * config.n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv

class RWKV_TinyAttn(nn.Module): # extra tiny attention
    def __init__(self, config):
        super().__init__()
        self.d_attn = config.rwkv_tiny_attn
        self.n_head = config.rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head

        self.qkv = nn.Linear(config.n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, config.n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)      # (B, T, C) -> (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))     # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim = -1)
        qkv = qk @ v                                                           # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)              # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
       
        return self.out(qkv)
    
class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = RWKV_TimeMix(config, layer_id)
        self.mlp = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x