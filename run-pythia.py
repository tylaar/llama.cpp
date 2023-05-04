from transformers import GPTNeoXForCausalLM, AutoTokenizer

import torch

dir_model = "/Users/yifengyu/hack/models"

torch.set_printoptions(sci_mode=False, linewidth=400)

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
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
print(tokenizer.decode(tokens[0]))
