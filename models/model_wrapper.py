# model_wrapper.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads)
        self.conv2 = GATv2Conv(hid_dim*heads, hid_dim, heads=1)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, data):
        # data.x: [N, feat_dim], data.edge_index
        if data is None or data.num_nodes == 0:
            return None
        h = self.conv1(data.x, data.edge_index)
        h = torch.relu(h)
        h = self.conv2(h, data.edge_index)
        h = torch.relu(h)
        pooled = global_mean_pool(h, data.batch)  # [B, hid_dim]
        return self.fc(pooled)  # [B, out_dim]

class GraphFusionCrossAttn(nn.Module):
    def __init__(self, text_dim, graph_dim, num_heads=8):
        super().__init__()
        self.to_k = nn.Linear(graph_dim, text_dim)
        self.to_v = nn.Linear(graph_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, kdim=text_dim, vdim=text_dim)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, text_hidden, graph_emb):
        # text_hidden: [B, seq_len, text_dim]
        # graph_emb: [B, graph_dim]  (we'll project)
        if graph_emb is None:
            return text_hidden
        k = self.to_k(graph_emb).unsqueeze(0)  # [1, B, D]
        v = self.to_v(graph_emb).unsqueeze(0)
        q = text_hidden.transpose(0,1)  # [seq, B, D]
        attn_out, _ = self.cross_attn(q, k, v)
        out = q + attn_out
        out = out.transpose(0,1)  # [B, seq, D]
        return self.norm(out)

class LlamaWithGraph(nn.Module):
    def __init__(self, llama_name, gnn_cfg):
        super().__init__()
        self.llama = AutoModelForCausalLM.from_pretrained(llama_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # freeze base -> we'll fine-tune via PEFT LoRA separately.
        for p in self.llama.parameters():
            p.requires_grad = False

        # GNNs: you must set in_dim to your graph node feature dim
        self.gnn_ast = SimpleGNN(in_dim=gnn_cfg['in_dim'], hid_dim=gnn_cfg['hid'], out_dim=gnn_cfg['out'])
        self.gnn_cfg = SimpleGNN(in_dim=gnn_cfg['in_dim_cfg'], hid_dim=gnn_cfg['hid'], out_dim=gnn_cfg['out'])
        self.gnn_dfg = SimpleGNN(in_dim=gnn_cfg['in_dim_dfg'], hid_dim=gnn_cfg['hid'], out_dim=gnn_cfg['out'])

        # fusion
        fused_dim = gnn_cfg['out'] * 3
        self.graph_project = nn.Linear(fused_dim, self.llama.config.hidden_size)
        self.cross_attn = GraphFusionCrossAttn(text_dim=self.llama.config.hidden_size, graph_dim=self.llama.config.hidden_size)

    def forward(self, input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
        # 1) LLaMA token embeddings -> pass through embedding layer to get inputs_embeds
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)
        # 2) compute graph embeddings
        emb_list = []
        for gnn, batch in [(self.gnn_ast, ast_batch), (self.gnn_cfg, cfg_batch), (self.gnn_dfg, dfg_batch)]:
            if batch is None:
                emb_list.append(torch.zeros(input_ids.size(0), gnn.fc.out_features, device=input_ids.device))
            else:
                emb = gnn(batch)  # [B, out_dim]
                emb_list.append(emb)
        fused = torch.cat(emb_list, dim=-1)  # [B, fused_dim]
        graph_ctx = self.graph_project(fused)  # [B, hidden_size]

        # 3) inject cross-attention
        # run initial LLaMA transformer forward but replace embeddings with cross-attn applied
        # simpler: call model with inputs_embeds but pre-process with cross_attn
        text_hidden = inputs_embeds  # [B, seq, D]
        text_hidden = self.cross_attn(text_hidden, graph_ctx)  # fuse graph context
        outputs = self.llama(inputs_embeds=text_hidden, attention_mask=attention_mask, labels=labels)
        return outputs