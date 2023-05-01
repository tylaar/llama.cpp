import os
import sys
import struct
import json
import torch
import numpy as np

dir_model = '/Users/yifengyu/hack/models/test'

fname_out = dir_model + "/test.bin";

fout = open(fname_out, "wb")
ftype = 0

fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
# todo: which vocab_sie is correct? in config.json or in tokenizer?
# vocab_size = hparams['vocab_size']
# vocab_size = len(vocab)
print('vocab_size:', 10)
fout.write(struct.pack("i", 10))

print('n_ctx:', 7)
fout.write(struct.pack("i", 7))

print('n_embd:', 6)
fout.write(struct.pack("i", 6))  # n_embd

print('n_head:', 2)
fout.write(struct.pack("i", 2))  # n_head

print('n_layer:', 1)
fout.write(struct.pack("i", 1))  # n_layer

fout.write(struct.pack("i", ftype))
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

state_dict["c_attn_q_w"] = c_attn_k_v_w[:, c_attn_q_idx].transpose()
state_dict["c_attn_k_w"] = c_attn_k_v_w[:, c_attn_k_idx].transpose()
state_dict["c_attn_v_w"] = c_attn_k_v_w[:, c_attn_v_idx].transpose()

c_attn_k_v_b = np.linspace(0.0, 0.000018, 18)
state_dict["c_attn_q_b"] = c_attn_k_v_b[c_attn_q_idx]
state_dict["c_attn_k_b"] = c_attn_k_v_b[c_attn_k_idx]
state_dict["c_attn_v_b"] = c_attn_k_v_b[c_attn_v_idx]

list_vars = state_dict

for name in list_vars.keys():
    data = list_vars[name]
    print("Processing variable: " + name + " with shape: ", data.shape)

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0;
    if ftype != 0:
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
