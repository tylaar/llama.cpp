# Convert GPT-J-6B h5 transformer model to ggml format
#
# Load the model using GPTJForCausalLM.
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#
import os
import sys
import struct
import json
import torch
import numpy as np

from transformers import GPTJForCausalLM, GPTNeoXForCausalLM, AutoTokenizer


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = chr
    bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    res = dict(zip(bs, cs))
    #todo hack
    res[' '] = 185
    return res

def calculate_rot_dim(hparams):
    base = hparams['rotary_emb_base']
    rotary_pct = hparams['rotary_pct']
    head_size = hparams['hidden_size'] / hparams['num_attention_heads']
    dim = int(head_size * rotary_pct)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq.shape[0]

if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    #sys.exit(1)

# output in the same directory as the model
dir_model = '/Users/yifengyu/hack/models/pythia-70m'
fname_out = dir_model + "/ggml-model.bin"

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

model = GPTNeoXForCausalLM.from_pretrained(
    dir_model,
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
    dir_model,
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
)

model_dict = torch.load(dir_model + '/pytorch_model.bin')

for k, v in model_dict.items():
    print("key:", k, " val:", v.shape)


try:
    os.remove(fname_out)
except OSError:
    pass

vocab = tokenizer.vocab
ftype = 1
fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
vocab_size = hparams['vocab_size']
print('vocab_size:', vocab_size)
fout.write(struct.pack("i", vocab_size))

n_head = hparams['num_attention_heads']
print('n_head:', n_head)
fout.write(struct.pack("i", n_head))

n_embd = hparams['hidden_size']
print('n_embd:', n_embd)
fout.write(struct.pack("i", n_embd))  # n_embd

n_head = hparams['num_attention_heads']
print('n_head:', n_head)
fout.write(struct.pack("i", n_head))  # n_head

n_layer = hparams['num_hidden_layers']
print('n_layer:', n_layer)
fout.write(struct.pack("i", n_layer))  # n_layer

rotary_dim = calculate_rot_dim(hparams)
print('rotary_dim:', rotary_dim)
fout.write(struct.pack("i", rotary_dim))
fout.write(struct.pack("i", ftype))

#f_vocab_out = open(dir_model + '/vocab.json', "wb")
#f_vocab_out.write(vocab_json)
#f_vocab_out.close()

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(vocab)))

for key in vocab:
    text = None
    if key.strip() == '':
        # TODO very big warning here.
        text = bytearray([185 for c in key])
    else:
        text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

list_vars = model.state_dict()

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    # we don't need these
    if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
        print("  Skipping variable: " + name)
        continue

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
