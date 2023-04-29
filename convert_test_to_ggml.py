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

print('n_ctx:', 5)
fout.write(struct.pack("i", 5))

print('n_embd:', 4)
fout.write(struct.pack("i", 4))  # n_embd

print('n_head:', 2)
fout.write(struct.pack("i", 2))  # n_head

print('n_layer:', 1)
fout.write(struct.pack("i", 1))  # n_layer

fout.write(struct.pack("i", ftype))
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
).transpose()
state_dict["c_attn_k_v_b"] = np.arange(0.0, 0.12, 0.01).transpose()

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
