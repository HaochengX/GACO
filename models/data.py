
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch
# Simple dataset that loads and flattens all samples from .pt files
class PTListDataset(Dataset):
    def __init__(self, pt_files):
        self.samples = []
        for f in pt_files:
            # NOTE: PyTorch 2.6 default is weights_only=True for safety; allow non-weight objects
            self.samples.extend(torch.load(f, weights_only=False))
        print(f"[Dataset] loaded {len(self.samples)} samples from {len(pt_files)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Collate: stack token tensors; build PyG Batch for each graph type
def collate_fn(batch):
    # tokens -> tensors
    batch = [x for x in batch if isinstance(x, dict) and "input_ids" in x]
    
    if not batch:
        raise ValueError("No valid items in batch")

    input_ids = torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"], dtype=torch.long) for x in batch])

    ast_list = [x["graph_ast"] for x in batch]
    cfg_list = [x["graph_cfg"] for x in batch]
    dfg_list = [x["graph_dfg"] for x in batch]

    ast_batch = PyGBatch.from_data_list(ast_list) if len(ast_list) else None
    cfg_batch = PyGBatch.from_data_list(cfg_list) if len(cfg_list) else None
    dfg_batch = PyGBatch.from_data_list(dfg_list) if len(dfg_list) else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ast_batch": ast_batch,
        "cfg_batch": cfg_batch,
        "dfg_batch": dfg_batch
    }


def list_pt_files(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "*.pt")))
