import torch

data = torch.load("processed_data/batch_0022.pt", weights_only=False)
sample = data[0]
print(len(sample))
print(sample["input_ids"])
# print(sample["graph"])    
print(sample["labels"])
print(sample["graph_ast"])
print(sample["graph_cfg"])
print(sample["graph_dfg"])