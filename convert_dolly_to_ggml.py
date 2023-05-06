import os
import sys
import struct
import json
import torch
import numpy as np

from transformers import GPTNeoXForCausalLM, AutoTokenizer

dir_model = "/Users/yifengyu/hack/models"

model = GPTNeoXForCausalLM.from_pretrained(
    dir_model + "/dolly-v2-3b",
    revision="step143000",
    cache_dir="./pythia-v2-3b/step143000",
    )

tokenizer = AutoTokenizer.from_pretrained(
    dir_model + "/dolly-v2-3b",
    revision="step143000",
    cache_dir="./dolly-v2-3b/step143000",
    )


def layer_transform(n_layer, list_vars, new_list_vars):
    c_attn_q_idx = np.array([], dtype=int)
    c_attn_k_idx = np.array([], dtype=int)
    c_attn_v_idx = np.array([], dtype=int)

    for i in range(n_head):
        c_attn_q_idx = np.concatenate([c_attn_q_idx, np.arange(i*3*head_size, i*3*head_size+head_size, 1, dtype=int)])
        c_attn_k_idx = c_attn_q_idx + n_embd/n_head
        c_attn_v_idx = c_attn_k_idx + n_embd/n_head


    for i in range(n_layer):
        istr = str(i)
        layer_norm_w = list_vars["gpt_neox.layers."+istr+".input_layernorm.weight"]
        new_list_vars["c_l_"+istr+"_norm_w"] = layer_norm_w
        layer_norm_b = list_vars["gpt_neox.layers."+istr+".input_layernorm.bias"]
        new_list_vars["c_l_"+istr+"_norm_b"] = layer_norm_b

        post_layer_norm_w = list_vars["gpt_neox.layers."+istr+".post_attention_layernorm.weight"]
        new_list_vars["c_l_"+istr+"_p_norm_w"] = post_layer_norm_w
        post_layer_norm_b = list_vars["gpt_neox.layers."+istr+".post_attention_layernorm.bias"]
        new_list_vars["c_l_"+istr+"_p_norm_b"] = post_layer_norm_b

        layer_dense_weight = list_vars["gpt_neox.layers."+istr+".attention.dense.weight"]
        new_list_vars["c_l_"+istr+"_dense_w"] = layer_dense_weight
        layer_dense_bias = list_vars["gpt_neox.layers."+istr+".attention.dense.bias"]
        new_list_vars["c_l_"+istr+"_dense_b"] = layer_dense_bias

        #
        # starting isolate out qkv part.
        #
        c_attn_k_v_w = list_vars["gpt_neox.layers."+istr+".attention.query_key_value.weight"]
        # c_attn_k_v_w = torch.transpose(c_attn_k_v_w, 0, 1)
        c_attn_k_v_b = list_vars["gpt_neox.layers."+istr+".attention.query_key_value.bias"]

        c_attn_q = c_attn_k_v_w[c_attn_q_idx, :]
        c_attn_k = c_attn_k_v_w[c_attn_k_idx, :]
        c_attn_v = c_attn_k_v_w[c_attn_v_idx, :]

        c_attn_q_b = c_attn_k_v_b[c_attn_q_idx]
        c_attn_k_b = c_attn_k_v_b[c_attn_k_idx]
        c_attn_v_b = c_attn_k_v_b[c_attn_v_idx]

        new_list_vars["c_l_"+istr+"_attn_k_w"] = c_attn_k
        new_list_vars["c_l_"+istr+"_attn_k_b"] = c_attn_k_b
        new_list_vars["c_l_"+istr+"_attn_q_w"] = c_attn_q
        new_list_vars["c_l_"+istr+"_attn_q_b"] = c_attn_q_b
        new_list_vars["c_l_"+istr+"_attn_v_w"] = c_attn_v
        new_list_vars["c_l_"+istr+"_attn_v_b"] = c_attn_v_b

        # End of isolate qkv part

        # start h_4h_h part
        c_mlp_h_4h_w = list_vars["gpt_neox.layers."+istr+".mlp.dense_h_to_4h.weight"]
        new_list_vars["c_l_"+istr+"_mlp_h_4h_w"] = c_mlp_h_4h_w
        c_mlp_h_4h_b = list_vars["gpt_neox.layers."+istr+".mlp.dense_h_to_4h.bias"]
        new_list_vars["c_l_"+istr+"_mlp_h_4h_b"] = c_mlp_h_4h_b
        c_mlp_4h_h_w = list_vars["gpt_neox.layers."+istr+".mlp.dense_4h_to_h.weight"]
        new_list_vars["c_l_"+istr+"_mlp_4h_h_w"] = c_mlp_4h_h_w
        c_mlp_4h_h_b = list_vars["gpt_neox.layers."+istr+".mlp.dense_4h_to_h.bias"]
        new_list_vars["c_l_"+istr+"_mlp_4h_h_b"] = c_mlp_4h_h_b
        # End of MLP
    return new_list_vars

dir_out_model = '/Users/yifengyu/hack/models/test'

fname_out = dir_out_model + "/test-dolly-v2-3b.bin"

fout = open(fname_out, "wb")

# 1.4B	1,208,602,624	24	2048	16	4M	2.0 x 10-4	GPT-Neo 1.3B, OPT-1.3B
fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
# todo: which vocab_sie is correct? in config.json or in tokenizer?
# vocab_size = hparams['vocab_size']
# vocab_size = len(vocab)
vocab_size = 50280
print('vocab_size:', vocab_size)
fout.write(struct.pack("i", vocab_size))

n_ctx = 512
print('n_ctx:', n_ctx)
fout.write(struct.pack("i", n_ctx))

n_embd = 2560
print('n_embd:', n_embd)
fout.write(struct.pack("i", n_embd))  # n_embd

n_head = 32
print('n_head:', n_head)
fout.write(struct.pack("i", n_head))  # n_head

n_layer = 32
print('n_layer:', n_layer)
fout.write(struct.pack("i", n_layer))  # n_layer
n_rot = int(n_embd / n_head / 4)
print("n_rot:", n_rot)
fout.write(struct.pack("i", n_rot))

ftype = 0
fout.write(struct.pack("i", ftype))

head_size = int(n_embd/n_head)

list_vars = model.state_dict()

new_list_vars = {}

final_norm_w = list_vars["gpt_neox.final_layer_norm.weight"]
new_list_vars["final_norm_w"] = final_norm_w
final_norm_b = list_vars["gpt_neox.final_layer_norm.bias"]
new_list_vars["final_norm_b"] = final_norm_b

embd_in_data = list_vars["gpt_neox.embed_in.weight"]
new_list_vars["embd_in"] = embd_in_data
embd_out_data = list_vars["embed_out.weight"]
new_list_vars["embd_out"] = embd_out_data

layer_transform(n_layer, list_vars, new_list_vars)

for idx, name in enumerate(new_list_vars):
    data = new_list_vars[name].numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0;
    if ftype != 0:
        # TODO: this is NOT fixed!
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # for efficiency - transpose these matrices:
    # (note - with latest ggml this is no longer more efficient, so disabling it)
    #  "transformer.h.*.mlp.fc_in.weight"
    #  "transformer.h.*.attn.out_proj.weight"
    #  "transformer.h.*.attn.q_proj.weight"
    #  "transformer.h.*.attn.k_proj.weight"
    #  "transformer.h.*.attn.v_proj.weight"
    #if name.endswith(".mlp.fc_in.weight")     or \
    #   name.endswith(".attn.out_proj.weight") or \
    #   name.endswith(".attn.q_proj.weight")   or \
    #   name.endswith(".attn.k_proj.weight")   or \
    #   name.endswith(".attn.v_proj.weight"):
    #    print("  Transposing")
    #    data = data.transpose()
    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
