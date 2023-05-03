import os
import sys
import struct
import json
import torch
import numpy as np
from torch import nn
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.activations import GELUActivation

dir_model = "/Users/yifengyu/hack/local_models"

model = GPTNeoXForCausalLM.from_pretrained(
    dir_model + "/pythia-70m",
    revision="step143000",
    cache_dir="./pythia-70m/step143000",
    )

tokenizer = AutoTokenizer.from_pretrained(
    dir_model + "/pythia-70m",
    revision="step143000",
    cache_dir="./pythia-70m/step143000",
    )

inputs = tokenizer("Hello, I am", return_tensors="pt")

def generate_diag_mask(max_ndims):
    z = torch.zeros((max_ndims, max_ndims))
    for i, x in enumerate(z):
        for j, y in enumerate(x):
            if i >= j:
                z[i][j] = 1.0
    return torch.gt(z, 0.0)

def print_2d_tensor_only_four(t):
    print("[", t[0][0], "...", t[0][-1], "]\n[", t[-1][0], "...", t[-1][-1], "]\n")

# Meta preparation stage.
vocab_size = 50304
# fout.write(struct.pack("i", 10))

# print('n_ctx:', 5)
n_ctx = 2048
# fout.write(struct.pack("i", 5))

# print('n_embd:', 4)
n_embd = 512
# fout.write(struct.pack("i", 4))  # n_embd

# print('n_head:', 2)
n_head = 8
# fout.write(struct.pack("i", 2))  # n_head

# print('n_layer:', 1)
n_layer = 1
# fout.write(struct.pack("i", 1))  # n_layer

# fout.write(struct.pack("i", ftype))
ftype = 0

norm_factor = torch.sqrt(torch.tensor(n_embd/n_head, dtype=torch.float32)).to(torch.get_default_dtype())

c_attn_q_idx = np.array([], dtype=int)
c_attn_k_idx = np.array([], dtype=int)
c_attn_v_idx = np.array([], dtype=int)

head_size = int(n_embd / n_head)

for i in range(n_head):
    c_attn_q_idx = np.concatenate([c_attn_q_idx, np.arange(i*3*head_size, i*3*head_size+head_size, 1, dtype=int)])
c_attn_k_idx = c_attn_q_idx + n_embd/n_head
c_attn_v_idx = c_attn_k_idx + n_embd/n_head

list_vars = model.state_dict()

layer_norm_w = list_vars["gpt_neox.layers.0.input_layernorm.weight"]
layer_norm_b = list_vars["gpt_neox.layers.0.input_layernorm.bias"]

post_layer_norm_w = list_vars["gpt_neox.layers.0.post_attention_layernorm.weight"]
post_layer_norm_b = list_vars["gpt_neox.layers.0.post_attention_layernorm.bias"]

layer_dense_weight = list_vars["gpt_neox.layers.0.attention.dense.weight"]
layer_dense_bias = list_vars["gpt_neox.layers.0.attention.dense.bias"]

c_attn_k_v_w = list_vars["gpt_neox.layers.0.attention.query_key_value.weight"]
c_attn_k_v_w = torch.transpose(c_attn_k_v_w, 0, 1)
c_attn_k_v_b = list_vars["gpt_neox.layers.0.attention.query_key_value.bias"]

c_attn_q = c_attn_k_v_w[:, c_attn_q_idx]
c_attn_k = c_attn_k_v_w[:, c_attn_k_idx]
c_attn_v = c_attn_k_v_w[:, c_attn_v_idx]
c_attn_q_b = c_attn_k_v_b[c_attn_q_idx]
c_attn_k_b = c_attn_k_v_b[c_attn_k_idx]
c_attn_v_b = c_attn_k_v_b[c_attn_v_idx]

embd_in_data = list_vars["gpt_neox.embed_in.weight"]
embd_out_data = list_vars["embed_out.weight"]

# starting to evaluate
embd_in = nn.Embedding(vocab_size, n_embd)
embd_in.weight.data = embd_in_data
embd_out = nn.Embedding(vocab_size, n_embd)
embd_out.weight.data = embd_out_data

layer_norm = nn.LayerNorm(n_embd)
layer_norm.weight.data = layer_norm_w
layer_norm.bias.data = layer_norm_b

post_layer_norm = nn.LayerNorm(n_embd)
post_layer_norm.weight.data = post_layer_norm_w
post_layer_norm.bias.data = post_layer_norm_b

c_h_to_4h_w = list_vars["gpt_neox.layers.0.mlp.dense_h_to_4h.weight"]
c_h_to_4h_b = list_vars["gpt_neox.layers.0.mlp.dense_h_to_4h.bias"]
c_4h_to_h_w = list_vars["gpt_neox.layers.0.mlp.dense_4h_to_h.weight"]
c_4h_to_h_b = list_vars["gpt_neox.layers.0.mlp.dense_4h_to_h.bias"]

nn_h_to_4h = nn.Linear(n_embd*4, n_embd)
nn_h_to_4h.weight.data = c_h_to_4h_w
nn_h_to_4h.bias.data = c_h_to_4h_b
nn_4h_to_h = nn.Linear(n_embd, n_embd*4)
nn_4h_to_h.weight.data = c_4h_to_h_w
nn_4h_to_h.bias.data = c_4h_to_h_b

gelu = GELUActivation()

nn_dense = nn.Linear(n_embd, n_embd)
nn_dense.weight.data = layer_dense_weight
nn_dense.bias.data = layer_dense_bias

embd_idx = torch.tensor([12092, 13, 309, 717])

selected_embd = embd_in(embd_idx)
selected_embd = layer_norm(selected_embd)
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

# head_size = int(n_embd/n_head)
new_qkv_shape = qkv.size()[:-1] + (n_head, 3 * head_size)
# c_attn_k_v_w_m = c_attn_k_v_w.view(*new_qkv_shape)
qkv = qkv.view(*new_qkv_shape)
print("=====qkv_reshaped printing======")
print(qkv)
print("=====qkv_reshaped printed=======")


query = qkv[..., : head_size].permute(0, 2, 1, 3)
key = qkv[..., head_size : 2 * head_size].permute(0, 2, 1, 3)
value = qkv[..., 2 * head_size :].permute(0, 2, 1, 3)

# print("normal query: \n", query)

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
akv_reshaped = torch.Size([4, n_head, head_size])
q_n = q.view(*akv_reshaped)
k_n = k.view(*akv_reshaped)
v_n = v.view(*akv_reshaped)

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
attn_output = torch.matmul(attn_weights, v_n_p[None, :])

# tensor: [bs, seq_len, hidden_size]
# -> [bs, seq_len, num_attention_heads, attn_head_size]
# tensor = attn_output.view(new_shape)
tensor = attn_output
print("*******************attn_output_merged_before_permute*******************")
print(tensor)
# -> [bs, num_attention_heads, seq_len, attn_head_size]
tensor = tensor.permute(0, 2, 1, 3)
print("*******************attn_output_merged_after_permute*******************")
print(tensor)

print("*******************attn_output_reshape_or_view_permute*******************")
tensor = tensor.reshape(tensor.size(0), tensor.size(1), num_attention_heads * head_size)
print(tensor)


print("*******************attn_densed*******************")
densed = nn_dense(tensor)
print(densed)
# do the _attn_ part
print("done")


print("*******************post_normed*******************")
post_normed = post_layer_norm(densed)
print(post_normed)
higher = nn_h_to_4h(post_normed)
gelued = gelu.forward(higher)
lower = nn_4h_to_h(gelued)
print(lower)


# do the _attn_ part
print("done")