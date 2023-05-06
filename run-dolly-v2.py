import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dir_model = "/Users/yifengyu/hack/models"

torch.set_printoptions(sci_mode=False, linewidth=400)

#tokenizer = AutoTokenizer.from_pretrained("./dolly-v2-3b", padding_side="left")
#model = AutoModelForCausalLM.from_pretrained("./dolly-v2-3b", device_map="auto", torch_dtype=torch.float32)

#inputs = tokenizer("Who is Obama?", return_tensors="pt")
#tokens = model.generate(**inputs)
#tokenizer.decode(tokens[0])
#print(tokenizer.decode(tokens[0]))
#import torch
from transformers import pipeline

generate_text = pipeline(model="/Users/yifengyu/hack/models/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

res = generate_text("Who is Obama?")
print(res[0]["generated_text"])
