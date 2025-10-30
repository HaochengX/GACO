# import pandas as pd

# splits = {'train': 'train.json', 'validation': 'valid.json'}
# df = pd.read_json("hf://datasets/likaixin/InstructCoder/" + splits["train"])

from datasets import load_dataset

dataset = load_dataset("likaixin/InstructCoder")
print(dataset["train"][0])

