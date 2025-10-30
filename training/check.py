import torch
from torch_geometric.data import Data

# Add safe globals for PyTorch Geometric
torch.serialization.add_safe_globals([Data])

# Load the data
data = torch.load("processed_data/training_data/processed_training_data.pt", weights_only=False)

# Check the data structure
print("Data type:", type(data))
print("Number of samples:", len(data))

# Get first sample
sample = data[0]
print("Sample keys:", list(sample.keys()))

# Convert to tensors if they're lists
input_ids = torch.tensor(sample['input_ids']) if isinstance(sample['input_ids'], list) else sample['input_ids']
labels = torch.tensor(sample['labels']) if isinstance(sample['labels'], list) else sample['labels']

print("Input IDs shape:", input_ids.shape)
print("Labels shape:", labels.shape)
print("Labels min/max:", labels.min().item(), labels.max().item())
print("Input IDs min/max:", input_ids.min().item(), input_ids.max().item())

# Check for NaN/Inf in the data
print("Labels has NaN:", torch.isnan(labels.float()).any().item())
print("Labels has Inf:", torch.isinf(labels.float()).any().item())
print("Input IDs has NaN:", torch.isnan(input_ids.float()).any().item())

# Check masking (corrected format validation)
masked_tokens = (labels == -100).sum().item()
total_tokens = labels.shape[0]
print(f"Masked tokens: {masked_tokens}/{total_tokens} ({masked_tokens/total_tokens*100:.1f}%)")

# Check if there are any non-masked tokens to train on
valid_tokens = (labels != -100).sum().item()
print(f"Valid training tokens: {valid_tokens}")

# Check graph data
if sample.get('graph_ast') is not None:
    print("AST graph available:", type(sample['graph_ast']))
    if hasattr(sample['graph_ast'], 'x'):
        print("AST graph features shape:", sample['graph_ast'].x.shape)
        print("AST features has NaN:", torch.isnan(sample['graph_ast'].x).any().item())
        print("AST features has Inf:", torch.isinf(sample['graph_ast'].x).any().item())
        print("AST features min/max:", sample['graph_ast'].x.min().item(), sample['graph_ast'].x.max().item())

if sample.get('graph_cfg') is not None:
    print("CFG graph available:", type(sample['graph_cfg']))
    if hasattr(sample['graph_cfg'], 'x'):
        print("CFG graph features shape:", sample['graph_cfg'].x.shape)
        print("CFG features has NaN:", torch.isnan(sample['graph_cfg'].x).any().item())

if sample.get('graph_dfg') is not None:
    print("DFG graph available:", type(sample['graph_dfg']))
    if hasattr(sample['graph_dfg'], 'x'):
        print("DFG graph features shape:", sample['graph_dfg'].x.shape)
        print("DFG features has NaN:", torch.isnan(sample['graph_dfg'].x).any().item())

# Check a few more samples for consistency
print("\nChecking multiple samples for NaN issues:")
for i in range(min(5, len(data))):
    sample = data[i]
    labels = torch.tensor(sample['labels']) if isinstance(sample['labels'], list) else sample['labels']
    has_nan = torch.isnan(labels.float()).any().item()
    valid_tokens = (labels != -100).sum().item()
    print(f"Sample {i}: NaN={has_nan}, valid_tokens={valid_tokens}")

masked_percentage = (labels == -100).float().mean() * 100
print(f"Masked: {masked_percentage:.1f}%")