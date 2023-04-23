# Convert MNIS h5 transformer model to ggml format
#
# Load the (state_dict) saved model using PyTorch
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# At the start of the ggml file we write the model parameters

import sys
import struct
import json
import numpy as np
import re


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

state_dict_file = "/Users/yifengyu/hack/models/mnist/mnist_model.state_dict"
fname_out = "/Users/yifengyu/hack/models/mnist/ggml-model-f32.bin"

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
#print (model)

list_vars = state_dict
print (list_vars)

fc2_weight = state_dict['fc2.weight']
print('0_0', fc2_weight[0][0])
print('1_0', fc2_weight[1][0])
print('9_0', fc2_weight[9][0])
print('0_1', fc2_weight[0][1])
print('0_9', fc2_weight[0][9])
print('5_250', fc2_weight[5][250])

print('debug finished')

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape) 
    n_dims = len(data.shape);
   
    fout.write(struct.pack("i", n_dims))
    
    data = data.astype(np.float32)
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
