import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GraphNorm, GATv2Conv, global_mean_pool
from transformers import AutoModelForCausalLM

GNN_HID = 256
GNN_OUT = 256                       # per-graph embedding dim (before fusion)
GRAPH_TOKEN_NUM = 128
GRAPH_HIDDEN_DIM = 768

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hid_dim=GNN_HID, out_dim=GNN_OUT, heads=4):
        super().__init__()
        
        # Initialize GATv2Conv with proper initialization
        self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads, dropout=0.1)
        self.norm1 = GraphNorm(hid_dim * heads)
        
        self.conv2 = GATv2Conv(hid_dim * heads, hid_dim, heads=1, dropout=0.1)
        self.norm2 = GraphNorm(hid_dim)
        
        # Add dropout layer that was missing
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(hid_dim, out_dim)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Better weight initialization to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Small positive bias
            elif isinstance(m, GraphNorm):
                # Initialize GraphNorm properly
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, data: PyGData):
        if data is None or data.num_nodes == 0:
            return None
            
        x, edge_index = data.x, data.edge_index

        # More aggressive NaN/Inf cleaning
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[GNN] Cleaning NaN/Inf in input features")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

        # First conv + norm + activation
        h = self.conv1(x, edge_index)
        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second conv + norm + activation
        h = self.conv2(h, edge_index)
        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        h = self.norm2(h)
        h = F.relu(h)
        
        # Global pooling with safety check
        if data.batch is None:
            # Single graph case
            pooled = torch.mean(h, dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(h, data.batch)
            
        # Final projection with gradient clipping
        output = self.fc(pooled)
        output = torch.clamp(output, -5.0, 5.0)  # Tighter bounds
        
        return output

class GraphFusionTokenGenerator(nn.Module):
    """
    Given per-graph pooled embeddings [B, out_dim] for AST/CFG/DFG,
    fuse them, project to graph_hidden_dim, and produce graph_token_num tokens per example.
    """
    def __init__(self, input_dim, graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        self.graph_token_num = graph_token_num
        self.graph_hidden_dim = graph_hidden_dim
        
        self.proj = nn.Linear(input_dim, graph_hidden_dim)
        # Initialize with smaller values
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.constant_(self.proj.bias, 0)
        
        # Smaller positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(graph_token_num, graph_hidden_dim) * 0.01)
        
        # Token MLP with proper initialization
        self.token_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        
        # Initialize MLP weights
        for m in self.token_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fused_graph_emb):  # fused_graph_emb: [B, input_dim]
        if fused_graph_emb is None:
            return None
            
        # Check for NaN/Inf
        if torch.isnan(fused_graph_emb).any() or torch.isinf(fused_graph_emb).any():
            print("[TokenGenerator] NaN/Inf in input!")
            fused_graph_emb = torch.nan_to_num(fused_graph_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
        B = fused_graph_emb.size(0)
        g = self.proj(fused_graph_emb)  # [B, graph_hidden_dim]
        
        # expand to tokens by repeating and adding pos emb
        tokens = g.unsqueeze(1).repeat(1, self.graph_token_num, 1)  # [B, graph_token_num, D]
        pos = self.pos_emb.unsqueeze(0).repeat(B, 1, 1)              # [B, graph_token_num, D]
        tokens = tokens + pos
        
        # Apply MLP with smaller scaling
        mlp_out = self.token_mlp(tokens)
        tokens = tokens + 0.1 * mlp_out  # Smaller residual contribution
        
        # Clamp output
        tokens = torch.clamp(tokens, -5, 5)
        return tokens

class GraphCrossAttention(nn.Module):
    def __init__(self, text_dim, graph_dim, num_heads=8):
        super().__init__()
        self.k_proj = nn.Linear(graph_dim, text_dim)
        self.v_proj = nn.Linear(graph_dim, text_dim)
        
        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, 
            num_heads=num_heads, 
            kdim=text_dim, 
            vdim=text_dim,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(text_dim, eps=1e-5)

    def forward(self, text_hidden, graph_tokens, attention_mask=None):
        # text_hidden: [B, seq_len, D], graph_tokens: [B, G, Dg] (projected)
        if graph_tokens is None:
            return text_hidden
            
        # Check for NaN/Inf
        if torch.isnan(graph_tokens).any() or torch.isinf(graph_tokens).any():
            print("[CrossAttention] NaN/Inf in graph tokens!")
            graph_tokens = torch.nan_to_num(graph_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # project graph tokens to text dim with smaller scaling
        k = self.k_proj(graph_tokens) * 0.1  # [B, G, D]
        v = self.v_proj(graph_tokens) * 0.1
        
        q = text_hidden.transpose(0, 1)   # [seq, B, D]
        k = k.transpose(0, 1)            # [G, B, D]
        v = v.transpose(0, 1)            # [G, B, D]
        
        # Ensure float32 for attention computation
        q = q.float()
        k = k.float()
        v = v.float()
        
        try:
            attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
        except RuntimeError as e:
            print(f"[CrossAttention] Error in attention: {e}")
            # Fallback: return original text hidden states
            return text_hidden
            
        # Small residual connection
        out = q + 0.1 * attn_out
        out = out.transpose(0, 1)  # [B, seq, D]
        out = out.to(text_hidden.dtype)
        return self.norm(out)

class LlamaWithGraph(nn.Module):
    def __init__(self, llama_path, tokenizer, gnn_in_dim_ast, gnn_in_dim_cfg, gnn_in_dim_dfg,
                 gnn_hid=GNN_HID, gnn_out=GNN_OUT, graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        print("[Model] loading base LLaMA (may take time)...")
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path, low_cpu_mem_usage=True, device_map=None, torch_dtype=torch.float16
        )
        for p in self.llama.parameters():
            p.requires_grad = False

        # CRITICAL: All GNN-related components must be in float32 to avoid dtype mismatch
        # Input projections with proper initialization - KEEP IN FLOAT32
        self.ast_in_proj = nn.Linear(gnn_in_dim_ast, gnn_in_dim_ast).float() if gnn_in_dim_ast > 0 else nn.Identity()
        self.cfg_in_proj = nn.Linear(gnn_in_dim_cfg, gnn_in_dim_cfg).float() if gnn_in_dim_cfg > 0 else nn.Identity()
        self.dfg_in_proj = nn.Linear(gnn_in_dim_dfg, gnn_in_dim_dfg).float() if gnn_in_dim_dfg > 0 else nn.Identity()
        
        # Initialize projections
        for proj in [self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight, gain=0.1)
                nn.init.constant_(proj.bias, 0)

        # GNNs in float32
        self.gnn_ast = SimpleGNN(in_dim=gnn_in_dim_ast, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_cfg = SimpleGNN(in_dim=gnn_in_dim_cfg, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_dfg = SimpleGNN(in_dim=gnn_in_dim_dfg, hid_dim=gnn_hid, out_dim=gnn_out).float()

        fused_in = gnn_out * 3
        self.fusion_proj = nn.Linear(fused_in, graph_hidden_dim).float()
        # Initialize fusion projection
        nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.1)
        nn.init.constant_(self.fusion_proj.bias, 0)

        self.token_generator = GraphFusionTokenGenerator(
            input_dim=graph_hidden_dim, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()

        self.cross_attn = GraphCrossAttention(
            text_dim=self.llama.config.hidden_size, graph_dim=graph_hidden_dim, num_heads=16
        ).float()
        
        # CRITICAL: Ensure all graph-related components are in float32
        self._ensure_graph_components_float32()
    
    def _ensure_graph_components_float32(self):
        """Ensure all graph processing components are in float32 to avoid dtype mismatches"""
        components_to_float32 = [
            self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj,
            self.gnn_ast, self.gnn_cfg, self.gnn_dfg,
            self.fusion_proj, self.token_generator, self.cross_attn
        ]
        
        for component in components_to_float32:
            if not isinstance(component, nn.Identity):
                component.float()
    def _prep_batch_for_gnn(self, batch: PyGData, proj: nn.Module, device: torch.device):
        if batch is None or getattr(batch, "num_nodes", 0) == 0:
            return None
        
        batch = batch.to(device)
        if batch.x is not None:
            # Ensure float32 for graph operations
            batch.x = batch.x.float()
            
            # More comprehensive cleaning
            batch.x = torch.where(
                torch.isnan(batch.x) | torch.isinf(batch.x) | (torch.abs(batch.x) > 100),
                torch.zeros_like(batch.x),
                batch.x
            )
            
            # Normalize features to prevent explosion
            if batch.x.numel() > 0:
                batch.x = torch.clamp(batch.x, -10.0, 10.0)
                # Optional: standardize features
                # batch.x = (batch.x - batch.x.mean(dim=0, keepdim=True)) / (batch.x.std(dim=0, keepdim=True) + 1e-8)

        # Apply input projection safely
        if not isinstance(proj, nn.Identity) and batch.x is not None:
            proj = proj.float()
            try:
                batch.x = proj(batch.x)
                batch.x = torch.clamp(batch.x, -10.0, 10.0)
            except Exception as e:
                print(f"[PrepGNN] Projection failed: {e}, using identity")
                pass
                
        return batch
        
    def forward(self, input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
        inputs_embeds = self.llama.get_input_embeddings()(input_ids)  # [B, seq, D]
        B = input_ids.size(0)
        dev = inputs_embeds.device

        # Prep GNN inputs
        ast_batch = self._prep_batch_for_gnn(ast_batch, self.ast_in_proj, dev)
        cfg_batch = self._prep_batch_for_gnn(cfg_batch, self.cfg_in_proj, dev)
        dfg_batch = self._prep_batch_for_gnn(dfg_batch, self.dfg_in_proj, dev)

        def safe_gnn(gnn, batch, graph_name):
            if batch is None or batch.num_nodes == 0:
                return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
            
            try:
                # Ensure the GNN is in float32 mode and on the right device
                gnn = gnn.float().to(dev)
                
                out = gnn(batch)
                if out is None:
                    print(f"[{graph_name}] GNN returned None, using zeros")
                    return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
                
                # Handle dimension mismatch
                if out.shape[0] != B:
                    print(f"[{graph_name}] Dimension mismatch: expected {B}, got {out.shape[0]}")
                    # Pad or truncate as needed
                    if out.shape[0] < B:
                        padding = torch.zeros((B - out.shape[0], out.shape[1]), device=dev, dtype=torch.float32)
                        out = torch.cat([out, padding], dim=0)
                    else:
                        out = out[:B]
                
                return torch.clamp(out, -10, 10)
            except Exception as e:
                print(f"[{graph_name}] GNN failed: {e}, using zeros")
                return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)

        # Always perform GNN computations in fp32 for stability
        # DISABLE autocast completely for GNN operations to prevent dtype issues
        emb_ast = safe_gnn(self.gnn_ast, ast_batch, "AST")
        emb_cfg = safe_gnn(self.gnn_cfg, cfg_batch, "CFG")
        emb_dfg = safe_gnn(self.gnn_dfg, dfg_batch, "DFG")

        # Fusion with safety checks - also keep in float32
        fused = torch.cat([emb_ast, emb_cfg, emb_dfg], dim=-1)
        
        # Check for NaN/Inf before fusion projection
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print("[Fusion] NaN/Inf detected before projection, cleaning...")
            fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure fusion_proj is in float32 and apply
        self.fusion_proj = self.fusion_proj.float().to(dev)
        fused = self.fusion_proj(fused)
        fused = torch.clamp(fused, -10, 10)

        # Generate graph tokens - keep token_generator in float32
        self.token_generator = self.token_generator.float().to(dev)
        graph_tokens = self.token_generator(fused)  # Already in fp32
        
        if graph_tokens is not None:
            # Apply tanh for stability and scale down
            graph_tokens = torch.tanh(graph_tokens) * 0.01
            graph_tokens = graph_tokens.to(inputs_embeds.dtype)  # Convert to model dtype

        # Cross-attention - ensure cross_attn handles dtype conversion properly
        self.cross_attn = self.cross_attn.float().to(dev)
        text_hidden = self.cross_attn(inputs_embeds, graph_tokens, attention_mask=attention_mask)

        # Final LLaMA pass
        outputs = self.llama(inputs_embeds=text_hidden, attention_mask=attention_mask, labels=labels)
        
        # Check for NaN in loss
        # if hasattr(outputs, 'loss') and torch.isnan(outputs.loss):
        if hasattr(outputs, 'loss') and outputs.loss is not None and torch.isnan(outputs.loss):
            print("[Model] NaN loss detected!")
            # You can add additional debugging here
        
            
        return outputs
    
    def save_graph_components(self, save_dir):
        """Save all graph-related components that are trainable"""
        graph_components = {
            'ast_in_proj': self.ast_in_proj.state_dict() if not isinstance(self.ast_in_proj, nn.Identity) else None,
            'cfg_in_proj': self.cfg_in_proj.state_dict() if not isinstance(self.cfg_in_proj, nn.Identity) else None,
            'dfg_in_proj': self.dfg_in_proj.state_dict() if not isinstance(self.dfg_in_proj, nn.Identity) else None,
            'gnn_ast': self.gnn_ast.state_dict(),
            'gnn_cfg': self.gnn_cfg.state_dict(),
            'gnn_dfg': self.gnn_dfg.state_dict(),
            'fusion_proj': self.fusion_proj.state_dict(),
            'token_generator': self.token_generator.state_dict(),
            'cross_attn': self.cross_attn.state_dict(),
        }
        
        # Save to file
        torch.save(graph_components, os.path.join(save_dir, "graph_components.pt"))
        print(f"[Save] Graph components saved to {save_dir}/graph_components.pt")
    
    def load_graph_components(self, save_dir):
        """Load all graph-related components"""
        graph_path = os.path.join(save_dir, "graph_components.pt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph components file not found: {graph_path}")
        
        graph_components = torch.load(graph_path, map_location='cpu')
        
        # Load each component
        if graph_components['ast_in_proj'] is not None:
            self.ast_in_proj.load_state_dict(graph_components['ast_in_proj'])
        if graph_components['cfg_in_proj'] is not None:
            self.cfg_in_proj.load_state_dict(graph_components['cfg_in_proj'])
        if graph_components['dfg_in_proj'] is not None:
            self.dfg_in_proj.load_state_dict(graph_components['dfg_in_proj'])
            
        self.gnn_ast.load_state_dict(graph_components['gnn_ast'])
        self.gnn_cfg.load_state_dict(graph_components['gnn_cfg'])
        self.gnn_dfg.load_state_dict(graph_components['gnn_dfg'])
        self.fusion_proj.load_state_dict(graph_components['fusion_proj'])
        self.token_generator.load_state_dict(graph_components['token_generator'])
        self.cross_attn.load_state_dict(graph_components['cross_attn'])
        
        print(f"[Load] Graph components loaded from {save_dir}/graph_components.pt")
