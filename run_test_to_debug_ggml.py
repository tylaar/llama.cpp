import os
import sys
import struct
import json
import torch
import numpy as np
from torch import nn

def generate_diag_mask(max_ndims):
    z = torch.zeros((max_ndims, max_ndims))
    for i, x in enumerate(z):
        for j, y in enumerate(x):
            if i >= j:
                z[i][j] = 1.0
    return torch.gt(z, 0.0)

# Meta preparation stage.
vocab_size = 10
# fout.write(struct.pack("i", 10))

# print('n_ctx:', 5)
n_ctx = 7
# fout.write(struct.pack("i", 5))

# print('n_embd:', 4)
n_embd = 6
# fout.write(struct.pack("i", 4))  # n_embd

# print('n_head:', 2)
n_head = 2
# fout.write(struct.pack("i", 2))  # n_head

# print('n_layer:', 1)
n_layer = 1
# fout.write(struct.pack("i", 1))  # n_layer

# fout.write(struct.pack("i", ftype))
ftype = 0

norm_factor = torch.sqrt(torch.tensor(n_embd/n_head, dtype=torch.float32)).to(torch.get_default_dtype())

# data preparation stage.
state_dict = {}

state_dict["embd_in"] = np.array(
    [
        np.linspace(0.01, 0.06, 6),  # a
        np.linspace(0.07, 0.12, 6),  # b
        np.linspace(0.13, 0.18, 6),  # c
        np.linspace(0.19, 0.24, 6),  # d
        np.linspace(0.25, 0.30, 6),  # e
        np.linspace(0.31, 0.36, 6),  # f
        np.linspace(0.37, 0.42, 6),  # g
        np.linspace(0.43, 0.48, 6),  # h
        np.linspace(0.49, 0.54, 6),  # i
        np.linspace(0.55, 0.60, 6),  # j
    ],
)

state_dict["embd_out"] = np.array(
    [
        np.linspace(0.01, 0.06, 6),  # a
        np.linspace(0.07, 0.12, 6),  # b
        np.linspace(0.13, 0.18, 6),  # c
        np.linspace(0.19, 0.24, 6),  # d
        np.linspace(0.25, 0.30, 6),  # e
        np.linspace(0.31, 0.36, 6),  # f
        np.linspace(0.37, 0.42, 6),  # g
        np.linspace(0.43, 0.48, 6),  # h
        np.linspace(0.49, 0.54, 6),  # i
        np.linspace(0.55, 0.60, 6),  # j
    ],
)

c_attn_q_idx = np.array([0,1,2,9,10,11]) # index
c_attn_k_idx = np.array([3,4,5,12,13,14]) #index
c_attn_v_idx = np.array([6,7,8,15,16,17]) # index

c_attn_k_v_w = np.array(
    [
        np.linspace(0.10, 0.27, 18),
        np.linspace(0.28, 0.45, 18),
        np.linspace(0.46, 0.63, 18),
        np.linspace(0.64, 0.81, 18),
        np.linspace(0.82, 0.99, 18),
        np.linspace(1.00, 1.17, 18),
    ]
)

state_dict["c_attn_k_v_w"] = c_attn_k_v_w

state_dict["c_attn_q_w"] = c_attn_k_v_w[:, c_attn_q_idx].transpose()
state_dict["c_attn_k_w"] = c_attn_k_v_w[:, c_attn_k_idx].transpose()
state_dict["c_attn_v_w"] = c_attn_k_v_w[:, c_attn_v_idx].transpose()

c_attn_k_v_b = np.linspace(0.0, 0.000018, 18)
state_dict["c_attn_k_v_b"] = c_attn_k_v_b

state_dict["c_attn_q_b"] = c_attn_k_v_b[c_attn_q_idx]
state_dict["c_attn_k_b"] = c_attn_k_v_b[c_attn_k_idx]
state_dict["c_attn_v_b"] = c_attn_k_v_b[c_attn_v_idx]

c_attn_k_v_w = torch.from_numpy(state_dict["c_attn_k_v_w"])
c_attn_k_v_b = torch.from_numpy(state_dict["c_attn_k_v_b"])

c_attn_q = c_attn_k_v_w[:, c_attn_q_idx]
c_attn_k = c_attn_k_v_w[:, c_attn_k_idx]
c_attn_v = c_attn_k_v_w[:, c_attn_v_idx]
c_attn_q_b = c_attn_k_v_b[c_attn_q_idx]
c_attn_k_b = c_attn_k_v_b[c_attn_k_idx]
c_attn_v_b = c_attn_k_v_b[c_attn_v_idx]

# starting to evaluate

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


query = qkv[..., : head_size].permute(0, 2, 1, 3)
key = qkv[..., head_size : 2 * head_size].permute(0, 2, 1, 3)
value = qkv[..., 2 * head_size :].permute(0, 2, 1, 3)

print("normal query: \n", query)

# TODO: below part is just trying to minic qkv separated into q, k, v
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
#print("self k:\n", k)
#print("self v:\n", v)
print("done")

qkv_permute = torch.Size([4, 2, 3])
q_n = q.view(*qkv_permute)
k_n = k.view(*qkv_permute)
v_n = v.view(*qkv_permute)
# print(q_n)

#print(q_n.permute(0, 2, 1))
#print(q_n.permute(1, 2, 0))
q_n_p = q_n.permute(1, 0, 2)
k_n_p = k_n.permute(1, 0, 2)
v_n_p = v_n.permute(1, 0, 2)
print("===============query_reshape_printing============")
print(q_n_p)
print("===============key_reshape_printing============")
print(k_n_p)
print("===============value_reshape_printing============")
print(v_n_p)

# starting to do the fucking attention here.
# starting to do the fucking attention here.
# starting to do the fucking attention here.

num_attention_heads, query_length, attn_head_size = q_n_p.size()
batch_size = 1
key_length = k_n_p.size(-2)

bias = generate_diag_mask(n_ctx)[None, None, :, :]

causal_mask = bias[:, :, key_length - query_length : key_length, :key_length]

print(causal_mask)

attn_scores = torch.zeros(
    batch_size * num_attention_heads,
    query_length,
    key_length,
    dtype=query.dtype,
    device=key.device,
    )

alpha = (torch.tensor(1.0, dtype=norm_factor.dtype, device=norm_factor.device) / norm_factor)

attn_scores = torch.baddbmm(
    attn_scores,
    q_n_p,
    k_n_p.transpose(1, 2),
    beta=1.0,
    alpha=alpha,
)
attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
print("*******************attn_score*******************")
print("===================score_before_causal mask=============")
print(attn_scores)
mask_value = torch.finfo(attn_scores.dtype).min
mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype)
print("===================score_after_causal mask=============")
attn_scores = torch.where(causal_mask, attn_scores, mask_value) # do the fucking mask

print(attn_scores)

# TODO: no attention_mask considered yet
# TODO: no head_mask considered yet.

print("*******************attn_weight*******************")
attn_weights = nn.functional.softmax(attn_scores, dim=-1)
print(attn_weights)
print("*******************attn_output*******************")
attn_output = torch.matmul(attn_weights, v_n_p[None, :])
print(attn_output);

# do the _attn_ part
print("done")