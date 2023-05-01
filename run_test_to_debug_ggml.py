import os
import sys
import struct
import json
import torch
import numpy as np
from torch import nn


# fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
# todo: which vocab_sie is correct? in config.json or in tokenizer?
# vocab_size = hparams['vocab_size']
# vocab_size = len(vocab)
# print('vocab_size:', 10)
vocab_size = 10
# fout.write(struct.pack("i", 10))

# print('n_ctx:', 5)
n_ctx = 5
# fout.write(struct.pack("i", 5))

# print('n_embd:', 4)
n_embd = 4
# fout.write(struct.pack("i", 4))  # n_embd

# print('n_head:', 2)
n_head = 2
# fout.write(struct.pack("i", 2))  # n_head

# print('n_layer:', 1)
n_layer = 1
# fout.write(struct.pack("i", 1))  # n_layer

# fout.write(struct.pack("i", ftype))
ftype = 0

state_dict = {}
state_dict["embd_in"] = np.array(
    [
        [0.01, 0.02, 0.03, 0.04],  # a
        [0.05, 0.06, 0.07, 0.08],  # b
        [0.09, 0.10, 0.11, 0.12],  # c
        [0.13, 0.14, 0.15, 0.16],  # d
        [0.17, 0.18, 0.19, 0.20],  # e
        [0.21, 0.22, 0.23, 0.24],  # f
        [0.25, 0.26, 0.27, 0.28],  # g
        [0.29, 0.30, 0.31, 0.32],  # h
        [0.33, 0.34, 0.35, 0.36],  # i
        [0.37, 0.38, 0.39, 0.40],  # j
    ])

state_dict["embd_out"] = np.array(
    [
        [0.01, 0.02, 0.03, 0.04],  # a
        [0.05, 0.06, 0.07, 0.08],  # b
        [0.09, 0.10, 0.11, 0.12],  # c
        [0.13, 0.14, 0.15, 0.16],  # d
        [0.17, 0.18, 0.19, 0.20],  # e
        [0.21, 0.22, 0.23, 0.24],  # f
        [0.25, 0.26, 0.27, 0.28],  # g
        [0.29, 0.30, 0.31, 0.32],  # h
        [0.33, 0.34, 0.35, 0.36],  # i
        [0.37, 0.38, 0.39, 0.40],  # j
    ])

state_dict["c_attn_k_v_w"] = np.array(
    [
        np.arange(0.1, 0.22, 0.01),
        np.arange(0.2, 0.32, 0.01),
        np.arange(0.3, 0.42, 0.01),
        np.arange(0.4, 0.52, 0.01),
    ]
)
state_dict["c_attn_k_v_b"] = np.arange(0.0, 0.000012, 0.000001)

c_attn_q_idx = torch.tensor([0,1,6,7])
c_attn_k_idx = torch.tensor([2,3,8,9])
c_attn_v_idx = torch.tensor([4,5,10,11])

c_attn_k_v_w = torch.from_numpy(state_dict["c_attn_k_v_w"])
c_attn_k_v_b = torch.from_numpy(state_dict["c_attn_k_v_b"])

c_attn_q = c_attn_k_v_w[:, c_attn_q_idx]
c_attn_k = c_attn_k_v_w[:, c_attn_k_idx]
c_attn_v = c_attn_k_v_w[:, c_attn_v_idx]
c_attn_q_b = c_attn_k_v_b[c_attn_q_idx]
c_attn_k_b = c_attn_k_v_b[c_attn_k_idx]
c_attn_v_b = c_attn_k_v_b[c_attn_v_idx]


e_in = torch.from_numpy(state_dict["embd_in"])
e_out = torch.from_numpy(state_dict["embd_out"])
embd_in = nn.Embedding(10, 4)
embd_in.weight.data = e_in
embd_out = nn.Embedding(10, 4)
embd_out.weight.data = e_out

embd_idx = torch.tensor([2,4,6,8])

selected_embd = embd_in(embd_idx)
print("=====original attn_k_v_w=====")
print(c_attn_k_v_w)
qkv = torch.add(
        torch.matmul(selected_embd, c_attn_k_v_w),
        c_attn_k_v_b
    )
print("=====added_and_malmuted_ attn_k_v_w=====")
qkv = qkv[None, :, :]
print(qkv)
print("=====added_and_malmuted_ attn_k_v_w shape=====")
print(qkv.shape)

head_size = int(n_embd/n_head)
new_qkv_shape = qkv.size()[:-1] + (n_head, 3 * head_size)
# c_attn_k_v_w_m = c_attn_k_v_w.view(*new_qkv_shape)
qkv = qkv.view(*new_qkv_shape)
print("=====qkv_reshaped printing======")
print(qkv)
print("=====qkv_reshaped printed=======")


query = qkv[..., : head_size]#.permute(0, 2, 1, 3)
key = qkv[..., head_size : 2 * head_size]#.permute(0, 2, 1, 3)
value = qkv[..., 2 * head_size :]#.permute(0, 2, 1, 3)

print(query, "\n", key, "\n", value, "\n")

q = torch.add(
    torch.matmul(selected_embd, c_attn_q),
    c_attn_q_b
)
k = torch.add(
    torch.matmul(selected_embd, c_attn_k),
    c_attn_k_b
)
v = torch.add(
    torch.matmul(selected_embd, c_attn_v),
    c_attn_v_b
)
print("self q:\n", q)
print("self k:\n", k)
print("self v:\n", v)
