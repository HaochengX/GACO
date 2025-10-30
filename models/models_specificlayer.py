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
        self.pos_emb = nn.Parameter(torch.randn(graph_token_num, graph_hidden_dim) * 0.1)
        
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

class GatedGraphCrossAttention(nn.Module):
    """
    Cross-attention with gated fusion to preserve original context
    """
    def __init__(self, text_dim, graph_dim, num_heads=8):
        super().__init__()
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        
        # Projection layers for graph tokens
        self.k_proj = nn.Linear(graph_dim, text_dim)
        self.v_proj = nn.Linear(graph_dim, text_dim)
        
        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        
        # Cross attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, 
            num_heads=num_heads, 
            kdim=text_dim, 
            vdim=text_dim,
            dropout=0.1
        )
        
        # Gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),  # Concat original + attended
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid()  # Gate values between 0 and 1
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(text_dim, eps=1e-5)
        
        # Initialize gate network to bias towards original context initially
        for i, layer in enumerate(self.gate_net):
            if isinstance(layer, nn.Linear):
                if i == len(self.gate_net) - 2:  # Second-to-last layer (before sigmoid)
                    # Initialize to produce values close to 0.5 after sigmoid
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.constant_(layer.bias, -0.5)  # Will give ~0.38 after sigmoid
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, text_hidden, graph_tokens, attention_mask=None):
        # text_hidden: [B, seq_len, D], graph_tokens: [B, G, Dg]
        if graph_tokens is None:
            return text_hidden
            
        # Check for NaN/Inf
        if torch.isnan(graph_tokens).any() or torch.isinf(graph_tokens).any():
            print("[GatedCrossAttention] NaN/Inf in graph tokens!")
            graph_tokens = torch.nan_to_num(graph_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
        
        B, seq_len, D = text_hidden.shape
        
        # Store original text hidden states
        text_original = text_hidden.clone()
        
        # Project graph tokens to text dim with smaller scaling
        k = self.k_proj(graph_tokens) # * 0.1  # [B, G, D]
        v = self.v_proj(graph_tokens) #* 0.1
        
        # Prepare for attention (seq_len first for PyTorch MultiheadAttention)
        q = text_hidden.transpose(0, 1)   # [seq, B, D]
        k = k.transpose(0, 1)            # [G, B, D]
        v = v.transpose(0, 1)            # [G, B, D]
        
        # Ensure float32 for attention computation
        q = q.float()
        k = k.float()
        v = v.float()
        
        try:
            # Cross attention
            attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
            attn_out = attn_out.transpose(0, 1)  # [B, seq, D]
            
            # Convert back to original dtype
            attn_out = attn_out.to(text_hidden.dtype)
            
            # Gated fusion
            # Concatenate original and attended representations
            concat_repr = torch.cat([text_original, attn_out], dim=-1)  # [B, seq, 2*D]
            
            # Compute gate values
            gate = self.gate_net(concat_repr)  # [B, seq, D]
            
            # Apply gating: gate * attended + (1 - gate) * original
            fused = gate * attn_out + (1 - gate) * text_original
            
            # Apply layer normalization
            out = self.norm(fused)
            
            return out
            
        except RuntimeError as e:
            print(f"[GatedCrossAttention] Error in attention: {e}")
            # Fallback: return original text hidden states
            return text_hidden

class LayerSpecificGraphIntegration(nn.Module):
    """
    Manages graph integration for specific layers only
    """
    def __init__(self, text_dim, graph_dim, target_layers=[0], num_heads=8):
        super().__init__()
        self.target_layers = set(target_layers)  # Convert to set for O(1) lookup
        
        # Create cross-attention modules for each target layer
        self.cross_attn = nn.ModuleDict({
            str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
            for layer in target_layers
        })
        
        print(f"[LayerSpecific] Graph integration enabled for layers: {sorted(target_layers)}")

    def should_apply_graph_integration(self, layer_idx):
        """Check if graph integration should be applied to this layer"""
        return layer_idx in self.target_layers
    
    def apply_graph_integration(self, layer_idx, text_hidden, graph_tokens, attention_mask=None):
        """Apply graph integration for the specified layer"""
        if not self.should_apply_graph_integration(layer_idx):
            return text_hidden
        
        layer_key = str(layer_idx)
        if layer_key in self.cross_attn:
            return self.cross_attn[layer_key](text_hidden, graph_tokens, attention_mask)
        else:
            print(f"[LayerSpecific] Warning: No cross-attention for layer {layer_idx}")
            return text_hidden

class LlamaWithGraphLayerSpecific(nn.Module):
    def __init__(self, llama_path, tokenizer, gnn_in_dim_ast, gnn_in_dim_cfg, gnn_in_dim_dfg,
                 target_layers=[0], gnn_hid=GNN_HID, gnn_out=GNN_OUT, 
                 graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        print("[Model] loading base LLaMA (may take time)...")
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path, low_cpu_mem_usage=True, device_map=None, torch_dtype=torch.float16
        )
        
        # Store target layers for graph integration
        self.target_layers = target_layers
        print(f"[Model] Graph integration will be applied to layers: {target_layers}")
        
        # Freeze LLaMA parameters
        for p in self.llama.parameters():
            p.requires_grad = False

        # Input projections - KEEP IN FLOAT32
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
        nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.1)
        nn.init.constant_(self.fusion_proj.bias, 0)

        self.token_generator = GraphFusionTokenGenerator(
            input_dim=graph_hidden_dim, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()

        self.graph_to_llama_proj = nn.Linear(graph_hidden_dim, 4096).float()
        nn.init.xavier_uniform_(self.graph_to_llama_proj.weight, gain=0.1)
        nn.init.constant_(self.graph_to_llama_proj.bias, 0)

        # Layer-specific graph integration
        self.graph_integration = LayerSpecificGraphIntegration(
            text_dim=self.llama.config.hidden_size, 
            graph_dim=graph_hidden_dim,
            target_layers=target_layers,
            num_heads=16
        ).float()
        
        # Hook into LLaMA layers for graph integration
        self._setup_layer_hooks()
        
        # Ensure all graph-related components are in float32
        self._ensure_graph_components_float32()
    
    def _ensure_graph_components_float32(self):
        """Ensure all graph processing components are in float32 to avoid dtype mismatches"""
        components_to_float32 = [
            self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj,
            self.gnn_ast, self.gnn_cfg, self.gnn_dfg,
            self.fusion_proj, self.token_generator, self.graph_integration,
            self.graph_to_llama_proj
        ]
        
        for component in components_to_float32:
            if not isinstance(component, nn.Identity):
                component.float()
    
    def _setup_layer_hooks(self):
        """Setup hooks to inject graph information into specific layers"""
        self.graph_tokens_cache = None  # Cache for graph tokens
        self.attention_mask_cache = None
        
        # Get the transformer layers
        if hasattr(self.llama, 'model') and hasattr(self.llama.model, 'layers'):
            layers = self.llama.model.layers
        elif hasattr(self.llama, 'transformer') and hasattr(self.llama.transformer, 'h'):
            layers = self.llama.transformer.h
        else:
            print("[Model] Warning: Could not find transformer layers for hooking")
            return
        
        # Register forward hooks for target layers
        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                layers[layer_idx].register_forward_hook(
                    self._create_layer_hook(layer_idx)
                )
                print(f"[Model] Registered hook for layer {layer_idx}")
    
    def _create_layer_hook(self, layer_idx):
        """Create a forward hook for a specific layer"""
        def layer_hook(module, input, output):
            # input[0] should be the hidden states
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                
                # Apply graph integration if we have cached graph tokens
                if self.graph_tokens_cache is not None:
                    enhanced_hidden = self.graph_integration.apply_graph_integration(
                        layer_idx, hidden_states, self.graph_tokens_cache, self.attention_mask_cache
                    )
                    
                    # Return modified output
                    if isinstance(output, tuple):
                        return (enhanced_hidden,) + output[1:]
                    else:
                        return enhanced_hidden
            
            return output
        
        return layer_hook
    
    def _prep_batch_for_gnn(self, batch: PyGData, proj: nn.Module, device: torch.device):
        if batch is None or getattr(batch, "num_nodes", 0) == 0:
            return None
        
        batch = batch.to(device)
        if batch.x is not None:
            batch.x = batch.x.float()
            
            batch.x = torch.where(
                torch.isnan(batch.x) | torch.isinf(batch.x) | (torch.abs(batch.x) > 100),
                torch.zeros_like(batch.x),
                batch.x
            )
            
            if batch.x.numel() > 0:
                batch.x = torch.clamp(batch.x, -10.0, 10.0)

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
        B = input_ids.size(0)
        dev = input_ids.device

        # Prep GNN inputs (same as before)
        ast_batch = self._prep_batch_for_gnn(ast_batch, self.ast_in_proj, dev)
        cfg_batch = self._prep_batch_for_gnn(cfg_batch, self.cfg_in_proj, dev)
        dfg_batch = self._prep_batch_for_gnn(dfg_batch, self.dfg_in_proj, dev)

        # GNN computations (same as before)
        def safe_gnn(gnn, batch, graph_name):
            if batch is None or batch.num_nodes == 0:
                return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
            
            try:
                gnn = gnn.float().to(dev)
                out = gnn(batch)
                if out is None:
                    return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
                
                if out.shape[0] != B:
                    if out.shape[0] < B:
                        padding = torch.zeros((B - out.shape[0], out.shape[1]), device=dev, dtype=torch.float32)
                        out = torch.cat([out, padding], dim=0)
                    else:
                        out = out[:B]
                
                return torch.clamp(out, -10, 10)
            except Exception as e:
                print(f"[{graph_name}] GNN failed: {e}, using zeros")
                return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)

        emb_ast = safe_gnn(self.gnn_ast, ast_batch, "AST")
        emb_cfg = safe_gnn(self.gnn_cfg, cfg_batch, "CFG")
        emb_dfg = safe_gnn(self.gnn_dfg, dfg_batch, "DFG")

        # Fusion with safety checks
        fused = torch.cat([emb_ast, emb_cfg, emb_dfg], dim=-1)
        
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.fusion_proj = self.fusion_proj.float().to(dev)
        fused = self.fusion_proj(fused)
        fused = torch.clamp(fused, -10, 10)

        # Generate graph tokens
        self.token_generator = self.token_generator.float().to(dev)
        graph_tokens = self.token_generator(fused)
        
        if graph_tokens is not None:
            # graph_tokens = torch.tanh(graph_tokens)
            
            # Project graph tokens to LLaMA embedding dimension
            self.graph_to_llama_proj = self.graph_to_llama_proj.float().to(dev)
            graph_tokens = self.graph_to_llama_proj(graph_tokens)  # Shape: [B, graph_token_num, llama_hidden_size]
            # print(f"[Model] Graph tokens shape after projection: {graph_tokens.shape}")

        # INSTEAD OF HOOKS: Modify inputs_embeds directly
        if graph_tokens is not None:
            # Get input embeddings
            inputs_embeds = self.llama.get_input_embeddings()(input_ids)
            
            # Convert graph tokens to same dtype as inputs_embeds
            graph_tokens = graph_tokens.to(inputs_embeds.dtype)
            
            # Simple concatenation or addition approach
            # Option 1: Prepend graph tokens
            seq_len = inputs_embeds.shape[1]
            graph_seq_len = graph_tokens.shape[1]
            
            # Extend attention mask
            extended_attention_mask = torch.cat([
                torch.ones(B, graph_seq_len, device=dev, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
            
            # Combine embeddings
            combined_embeds = torch.cat([graph_tokens, inputs_embeds], dim=1)
            
            # Adjust labels if provided
            if labels is not None:
                # Pad labels with -100 for graph tokens (ignore in loss)
                graph_labels = torch.full((B, graph_seq_len), -100, device=dev, dtype=labels.dtype)
                extended_labels = torch.cat([graph_labels, labels], dim=1)
            else:
                extended_labels = None
            
            # Forward through LLaMA with modified inputs
            outputs = self.llama(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
        else:
            # Standard forward pass without graph integration
            outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs  

    def save_graph_components(self, save_dir):
        """Save all graph-related components that are trainable"""
        graph_components = {
            'target_layers': self.target_layers,
            'ast_in_proj': self.ast_in_proj.state_dict() if not isinstance(self.ast_in_proj, nn.Identity) else None,
            'cfg_in_proj': self.cfg_in_proj.state_dict() if not isinstance(self.cfg_in_proj, nn.Identity) else None,
            'dfg_in_proj': self.dfg_in_proj.state_dict() if not isinstance(self.dfg_in_proj, nn.Identity) else None,
            'gnn_ast': self.gnn_ast.state_dict(),
            'gnn_cfg': self.gnn_cfg.state_dict(),
            'gnn_dfg': self.gnn_dfg.state_dict(),
            'fusion_proj': self.fusion_proj.state_dict(),
            'token_generator': self.token_generator.state_dict(),
            'graph_integration': self.graph_integration.state_dict(),
        }
        
        torch.save(graph_components, os.path.join(save_dir, "graph_components_layerwise.pt"))
        print(f"[Save] Graph components (layer-wise) saved to {save_dir}/graph_components_layerwise.pt")
    
    def load_graph_components(self, save_dir):
        """Load all graph-related components"""
        graph_path = os.path.join(save_dir, "graph_components_layerwise.pt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph components file not found: {graph_path}")
        
        graph_components = torch.load(graph_path, map_location='cpu')
        
        # Check if target layers match
        if 'target_layers' in graph_components and graph_components['target_layers'] != self.target_layers:
            print(f"[Load] Warning: Target layers mismatch. Saved: {graph_components['target_layers']}, Current: {self.target_layers}")
        
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
        self.graph_integration.load_state_dict(graph_components['graph_integration'])
        
        print(f"[Load] Graph components (layer-wise) loaded from {save_dir}/graph_components_layerwise.pt")

# import torch
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data as PyGData
# from torch_geometric.nn import GraphNorm, GATv2Conv, global_mean_pool
# from transformers import AutoModelForCausalLM

# GNN_HID = 128
# GNN_OUT = 128  
# GRAPH_TOKEN_NUM = 128
# GRAPH_HIDDEN_DIM = 768
# class SimpleGNN(nn.Module):
#     def __init__(self, in_dim, hid_dim=GNN_HID, out_dim=GNN_OUT, heads=4):
#         super().__init__()
        
#         # Initialize GATv2Conv with proper initialization
#         self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads, dropout=0.1)
#         self.norm1 = GraphNorm(hid_dim * heads)
        
#         self.conv2 = GATv2Conv(hid_dim * heads, hid_dim, heads=1, dropout=0.1)
#         self.norm2 = GraphNorm(hid_dim)
        
#         # Add dropout layer that was missing
#         self.dropout = nn.Dropout(0.1)
        
#         self.fc = nn.Linear(hid_dim, out_dim)
        
#         # Initialize weights properly
#         self._init_weights()

#     def _init_weights(self):
#         """Fixed weight initialization for gradient stability"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 # FIXED: Use consistent xavier initialization instead of kaiming
#                 nn.init.xavier_uniform_(m.weight, gain=0.1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)  # Zero bias instead of 0.01
#             elif isinstance(m, GraphNorm):
#                 # Initialize GraphNorm properly
#                 if hasattr(m, 'weight') and m.weight is not None:
#                     nn.init.constant_(m.weight, 1.0)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, data: PyGData):
#         if data is None or data.num_nodes == 0:
#             return None
            
#         x, edge_index = data.x, data.edge_index

#         # More aggressive NaN/Inf cleaning
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print(f"[GNN] Cleaning NaN/Inf in input features")
#             x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

#         # First conv + norm + activation
#         h = self.conv1(x, edge_index)
#         h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
#         h = self.norm1(h)
#         h = F.relu(h)
#         h = self.dropout(h)
        
#         # Second conv + norm + activation
#         h = self.conv2(h, edge_index)
#         h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
#         h = self.norm2(h)
#         h = F.relu(h)
        
#         # Global pooling with safety check
#         if data.batch is None:
#             # Single graph case
#             pooled = torch.mean(h, dim=0, keepdim=True)
#         else:
#             pooled = global_mean_pool(h, data.batch)
            
#         # FIXED: Less aggressive clamping to preserve gradients
#         output = self.fc(pooled)
#         output = torch.clamp(output, -1.0, 1.0)  # Changed from -5,5 to -1,1
        
#         return output

# class GraphFusionTokenGenerator(nn.Module):
#     """
#     Given per-graph pooled embeddings [B, out_dim] for AST/CFG/DFG,
#     fuse them, project to graph_hidden_dim, and produce graph_token_num tokens per example.
#     """
#     def __init__(self, input_dim, graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
#         super().__init__()
#         self.graph_token_num = graph_token_num
#         self.graph_hidden_dim = graph_hidden_dim
        
#         self.proj = nn.Linear(input_dim, graph_hidden_dim)
#         # Initialize with smaller values
#         nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
#         nn.init.constant_(self.proj.bias, 0)
        
#         # Smaller positional embeddings
#         self.pos_emb = nn.Parameter(torch.randn(graph_token_num, graph_hidden_dim) * 0.01)
        
#         # Token MLP with proper initialization
#         self.token_mlp = nn.Sequential(
#             nn.Linear(graph_hidden_dim, graph_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(graph_hidden_dim, graph_hidden_dim)
#         )
        
#         # Initialize MLP weights
#         for m in self.token_mlp:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=0.1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, fused_graph_emb):  # fused_graph_emb: [B, input_dim]
#         if fused_graph_emb is None:
#             return None
            
#         # Check for NaN/Inf
#         if torch.isnan(fused_graph_emb).any() or torch.isinf(fused_graph_emb).any():
#             print("[TokenGenerator] NaN/Inf in input!")
#             fused_graph_emb = torch.nan_to_num(fused_graph_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
#         B = fused_graph_emb.size(0)
#         g = self.proj(fused_graph_emb)  # [B, graph_hidden_dim]
        
#         # expand to tokens by repeating and adding pos emb
#         tokens = g.unsqueeze(1).repeat(1, self.graph_token_num, 1)  # [B, graph_token_num, D]
#         pos = self.pos_emb.unsqueeze(0).repeat(B, 1, 1)              # [B, graph_token_num, D]
#         tokens = tokens + pos
        
#         # Apply MLP with smaller scaling
#         mlp_out = self.token_mlp(tokens)
#         tokens = tokens + 0.01 * mlp_out  # Smaller residual contribution
        
#         # Clamp output
#         tokens = torch.clamp(tokens, -5, 5)
#         return tokens

# class GatedGraphCrossAttention(nn.Module):
#     """
#     Cross-attention with gated fusion to preserve original context
#     """
#     def __init__(self, text_dim, graph_dim, num_heads=8):
#         super().__init__()
#         self.text_dim = text_dim
#         self.graph_dim = graph_dim
        
#         # Projection layers for graph tokens
#         self.k_proj = nn.Linear(graph_dim, text_dim)
#         self.v_proj = nn.Linear(graph_dim, text_dim)
        
#         # Initialize with smaller weights
#         nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
#         nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
#         nn.init.constant_(self.k_proj.bias, 0)
#         nn.init.constant_(self.v_proj.bias, 0)
        
#         # Cross attention module
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=text_dim, 
#             num_heads=num_heads, 
#             kdim=text_dim, 
#             vdim=text_dim,
#             dropout=0.1
#         )
        
#         # Gating mechanism
#         self.gate_net = nn.Sequential(
#             nn.Linear(text_dim * 2, text_dim),  # Concat original + attended
#             nn.ReLU(),
#             nn.Linear(text_dim, text_dim),
#             nn.Sigmoid()  # Gate values between 0 and 1
#         )
        
#         # Layer normalization
#         self.norm = nn.LayerNorm(text_dim, eps=1e-5)
        
#         # Initialize gate network to bias towards original context initially
#         for i, layer in enumerate(self.gate_net):
#             if isinstance(layer, nn.Linear):
#                 if i == len(self.gate_net) - 2:  # Second-to-last layer (before sigmoid)
#                     # Initialize to produce values close to 0.5 after sigmoid
#                     nn.init.xavier_uniform_(layer.weight, gain=0.1)
#                     nn.init.constant_(layer.bias, -0.5)  # Will give ~0.38 after sigmoid
#                 else:
#                     nn.init.xavier_uniform_(layer.weight, gain=0.1)
#                     nn.init.constant_(layer.bias, 0)

#     def forward(self, text_hidden, graph_tokens, attention_mask=None):
#         # text_hidden: [B, seq_len, D], graph_tokens: [B, G, Dg]
#         if graph_tokens is None:
#             return text_hidden
            
#         # Check for NaN/Inf
#         if torch.isnan(graph_tokens).any() or torch.isinf(graph_tokens).any():
#             print("[GatedCrossAttention] NaN/Inf in graph tokens!")
#             graph_tokens = torch.nan_to_num(graph_tokens, nan=0.0, posinf=1.0, neginf=-1.0)
        
#         B, seq_len, D = text_hidden.shape
        
#         # Store original text hidden states
#         text_original = text_hidden.clone()
        
#         # Project graph tokens to text dim with smaller scaling
#         k = self.k_proj(graph_tokens) * 0.1  # [B, G, D]
#         v = self.v_proj(graph_tokens) * 0.1
        
#         # Prepare for attention (seq_len first for PyTorch MultiheadAttention)
#         q = text_hidden.transpose(0, 1)   # [seq, B, D]
#         k = k.transpose(0, 1)            # [G, B, D]
#         v = v.transpose(0, 1)            # [G, B, D]
        
#         # Ensure float32 for attention computation
#         q = q.float()
#         k = k.float()
#         v = v.float()
        
#         try:
#             # Cross attention
#             attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
#             attn_out = attn_out.transpose(0, 1)  # [B, seq, D]
            
#             # Convert back to original dtype
#             attn_out = attn_out.to(text_hidden.dtype)
            
#             # Gated fusion
#             # Concatenate original and attended representations
#             concat_repr = torch.cat([text_original, attn_out], dim=-1)  # [B, seq, 2*D]
            
#             # Compute gate values
#             gate = self.gate_net(concat_repr)  # [B, seq, D]
            
#             # Apply gating: gate * attended + (1 - gate) * original
#             fused = gate * attn_out + (1 - gate) * text_original
            
#             # Apply layer normalization
#             out = self.norm(fused)
            
#             return out
            
#         except RuntimeError as e:
#             print(f"[GatedCrossAttention] Error in attention: {e}")
#             # Fallback: return original text hidden states
#             return text_hidden

# class LayerSpecificGraphIntegration(nn.Module):
#     """
#     Manages graph integration for specific layers only
#     """
#     def __init__(self, text_dim, graph_dim, target_layers=[0], num_heads=8):
#         super().__init__()
#         self.target_layers = set(target_layers)  # Convert to set for O(1) lookup
        
#         # Create cross-attention modules for each target layer
#         self.cross_attn = nn.ModuleDict({
#             str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
#             for layer in target_layers
#         })
        
#         print(f"[LayerSpecific] Graph integration enabled for layers: {sorted(target_layers)}")

#     def should_apply_graph_integration(self, layer_idx):
#         """Check if graph integration should be applied to this layer"""
#         return layer_idx in self.target_layers
    
#     def apply_graph_integration(self, layer_idx, text_hidden, graph_tokens, attention_mask=None):
#         """Apply graph integration for the specified layer"""
#         if not self.should_apply_graph_integration(layer_idx):
#             return text_hidden
        
#         layer_key = str(layer_idx)
#         if layer_key in self.cross_attn:
#             return self.cross_attn[layer_key](text_hidden, graph_tokens, attention_mask)
#         else:
#             print(f"[LayerSpecific] Warning: No cross-attention for layer {layer_idx}")
#             return text_hidden

# class LlamaWithGraphLayerSpecific(nn.Module):
#     def __init__(self, llama_path, tokenizer, gnn_in_dim_ast, gnn_in_dim_cfg, gnn_in_dim_dfg,
#                  target_layers=[0], gnn_hid=GNN_HID, gnn_out=GNN_OUT, 
#                  graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
#         super().__init__()
#         print("[Model] loading base LLaMA (may take time)...")
#         self.llama = AutoModelForCausalLM.from_pretrained(
#             llama_path, low_cpu_mem_usage=True, device_map=None, torch_dtype=torch.float16
#         )
        
#         # Store target layers for graph integration
#         self.target_layers = target_layers
#         print(f"[Model] Graph integration will be applied to layers: {target_layers}")
        
#         # Freeze LLaMA parameters
#         for p in self.llama.parameters():
#             p.requires_grad = False

#         # Input projections - KEEP IN FLOAT32
#         self.ast_in_proj = nn.Linear(gnn_in_dim_ast, gnn_in_dim_ast).float() if gnn_in_dim_ast > 0 else nn.Identity()
#         self.cfg_in_proj = nn.Linear(gnn_in_dim_cfg, gnn_in_dim_cfg).float() if gnn_in_dim_cfg > 0 else nn.Identity()
#         self.dfg_in_proj = nn.Linear(gnn_in_dim_dfg, gnn_in_dim_dfg).float() if gnn_in_dim_dfg > 0 else nn.Identity()
        
#         # Initialize projections
#         for proj in [self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj]:
#             if isinstance(proj, nn.Linear):
#                 nn.init.xavier_uniform_(proj.weight, gain=0.1)
#                 nn.init.constant_(proj.bias, 0)

#         # GNNs in float32
#         self.gnn_ast = SimpleGNN(in_dim=gnn_in_dim_ast, hid_dim=gnn_hid, out_dim=gnn_out).float()
#         self.gnn_cfg = SimpleGNN(in_dim=gnn_in_dim_cfg, hid_dim=gnn_hid, out_dim=gnn_out).float()
#         self.gnn_dfg = SimpleGNN(in_dim=gnn_in_dim_dfg, hid_dim=gnn_hid, out_dim=gnn_out).float()

#         fused_in = gnn_out * 3
#         self.fusion_proj = nn.Linear(fused_in, graph_hidden_dim).float()
#         nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.1)
#         nn.init.constant_(self.fusion_proj.bias, 0)

#         self.token_generator = GraphFusionTokenGenerator(
#             input_dim=graph_hidden_dim, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
#         ).float()
#         self.graph_to_llama_proj = nn.Linear(graph_hidden_dim, 4096).float()
#         # self.graph_to_llama_proj = self.graph_to_llama_proj.float().to(dev)

#         nn.init.xavier_uniform_(self.graph_to_llama_proj.weight, gain=0.01)
#         nn.init.constant_(self.graph_to_llama_proj.bias, 0)
#         # self.graph_to_llama_proj = nn.Sequential(
#         #     nn.Linear(graph_hidden_dim, 1024),
#         #     nn.GELU(),
#         #     nn.Linear(1024, 4096),
#         #     nn.LayerNorm(4096)   # stabilize before concat
#         # )
#         # for m in self.graph_to_llama_proj:
#         #     if isinstance(m, nn.Linear):
#         #         nn.init.xavier_uniform_(m.weight, gain=1e-3)
#         #         nn.init.constant_(m.bias, 0.0)

#         # Layer-specific graph integration
#         self.graph_integration = LayerSpecificGraphIntegration(
#             text_dim=self.llama.config.hidden_size, 
#             graph_dim=graph_hidden_dim,
#             target_layers=target_layers,
#             num_heads=16
#         ).float()
        
#         # Hook into LLaMA layers for graph integration
#         self._setup_layer_hooks()
        
#         # Ensure all graph-related components are in float32
#         self._ensure_graph_components_float32()
    
#     def _ensure_graph_components_float32(self):
#         """Ensure all graph processing components are in float32 to avoid dtype mismatches"""
#         components_to_float32 = [
#             self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj,
#             self.gnn_ast, self.gnn_cfg, self.gnn_dfg,
#             self.fusion_proj, self.token_generator, self.graph_integration,
#             self.graph_to_llama_proj
#         ]
        
#         for component in components_to_float32:
#             if not isinstance(component, nn.Identity):
#                 component.float()
    
#     def _setup_layer_hooks(self):
#         """Setup hooks to inject graph information into specific layers"""
#         self.graph_tokens_cache = None  # Cache for graph tokens
#         self.attention_mask_cache = None
        
#         # Get the transformer layers
#         if hasattr(self.llama, 'model') and hasattr(self.llama.model, 'layers'):
#             layers = self.llama.model.layers
#         elif hasattr(self.llama, 'transformer') and hasattr(self.llama.transformer, 'h'):
#             layers = self.llama.transformer.h
#         else:
#             print("[Model] Warning: Could not find transformer layers for hooking")
#             return
        
#         # Register forward hooks for target layers
#         for layer_idx in self.target_layers:
#             if layer_idx < len(layers):
#                 layers[layer_idx].register_forward_hook(
#                     self._create_layer_hook(layer_idx)
#                 )
#                 print(f"[Model] Registered hook for layer {layer_idx}")
    
#     def _create_layer_hook(self, layer_idx):
#         """Create a forward hook for a specific layer"""
#         def layer_hook(module, input, output):
#             # input[0] should be the hidden states
#             if isinstance(input, tuple) and len(input) > 0:
#                 hidden_states = input[0]
                
#                 # Apply graph integration if we have cached graph tokens
#                 if self.graph_tokens_cache is not None:
#                     enhanced_hidden = self.graph_integration.apply_graph_integration(
#                         layer_idx, hidden_states, self.graph_tokens_cache, self.attention_mask_cache
#                     )
                    
#                     # Return modified output
#                     if isinstance(output, tuple):
#                         return (enhanced_hidden,) + output[1:]
#                     else:
#                         return enhanced_hidden
            
#             return output
        
#         return layer_hook
    
#     def _prep_batch_for_gnn(self, batch: PyGData, proj: nn.Module, device: torch.device):
#         if batch is None or getattr(batch, "num_nodes", 0) == 0:
#             return None
        
#         batch = batch.to(device)
#         if batch.x is not None:
#             batch.x = batch.x.float()
            
#             batch.x = torch.where(
#                 torch.isnan(batch.x) | torch.isinf(batch.x) | (torch.abs(batch.x) > 100),
#                 torch.zeros_like(batch.x),
#                 batch.x
#             )
            
#             if batch.x.numel() > 0:
#                 batch.x = torch.clamp(batch.x, -10.0, 10.0)

#         if not isinstance(proj, nn.Identity) and batch.x is not None:
#             proj = proj.float()
#             try:
#                 batch.x = proj(batch.x)
#                 batch.x = torch.clamp(batch.x, -10.0, 10.0)
#             except Exception as e:
#                 print(f"[PrepGNN] Projection failed: {e}, using identity")
#                 pass
                
#         return batch

#     def forward(self, input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
#         B = input_ids.size(0)
#         dev = input_ids.device

#         # Prep GNN inputs (same as before)
#         ast_batch = self._prep_batch_for_gnn(ast_batch, self.ast_in_proj, dev)
#         cfg_batch = self._prep_batch_for_gnn(cfg_batch, self.cfg_in_proj, dev)
#         dfg_batch = self._prep_batch_for_gnn(dfg_batch, self.dfg_in_proj, dev)

#         # GNN computations (same as before)
#         def safe_gnn(gnn, batch, graph_name):
#             if batch is None or batch.num_nodes == 0:
#                 return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
            
#             try:
#                 gnn = gnn.float().to(dev)
#                 out = gnn(batch)
#                 if out is None:
#                     return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
                
#                 if out.shape[0] != B:
#                     if out.shape[0] < B:
#                         padding = torch.zeros((B - out.shape[0], out.shape[1]), device=dev, dtype=torch.float32)
#                         out = torch.cat([out, padding], dim=0)
#                     else:
#                         out = out[:B]
                
#                 return torch.clamp(out, -10, 10)
#             except Exception as e:
#                 print(f"[{graph_name}] GNN failed: {e}, using zeros")
#                 return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)

#         emb_ast = safe_gnn(self.gnn_ast, ast_batch, "AST")
#         emb_cfg = safe_gnn(self.gnn_cfg, cfg_batch, "CFG")
#         emb_dfg = safe_gnn(self.gnn_dfg, dfg_batch, "DFG")

#         # Fusion with safety checks
#         fused = torch.cat([emb_ast, emb_cfg, emb_dfg], dim=-1)
        
#         if torch.isnan(fused).any() or torch.isinf(fused).any():
#             fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        
#         # self.fusion_proj = self.fusion_proj.float().to(dev)
#         fused = self.fusion_proj(fused)
#         fused = torch.clamp(fused, -10, 10)

#         # Generate graph tokens
#         # self.token_generator = self.token_generator.float().to(dev)
#         graph_tokens = self.token_generator(fused)

#         # with torch.cuda.amp.autocast(enabled=False):
#         #     graph_tokens = self.graph_to_llama_proj(graph_tokens.float())
#         # graph_tokens = 0.001 * F.layer_norm(graph_tokens, graph_tokens.shape[-1:])
        


#         # graph_tokens = graph_tokens.to(inputs_embeds.dtype)

#         # graph_tokens = self.graph_to_llama_proj(graph_tokens.float())
#         # graph_tokens = 0.01 * graph_tokens      # shrink scale
#         # graph_tokens = graph_tokens.to(inputs_embeds.dtype)
#         if graph_tokens is not None:
#             # graph_tokens = torch.tanh(graph_tokens) 
#             # graph_tokens = 0.01 * graph_tokens 
#             # Project graph tokens to LLaMA embedding dimension
#             # self.graph_to_llama_proj = self.graph_to_llama_proj.float().to(dev)
#             graph_tokens = self.graph_to_llama_proj(graph_tokens)  # Shape: [B, graph_token_num, llama_hidden_size]
#             # print(f"[Model] Graph tokens shape after projection: {graph_tokens.shape}")

#         # INSTEAD OF HOOKS: Modify inputs_embeds directly
#         if graph_tokens is not None:
#             # Get input embeddings
#             inputs_embeds = self.llama.get_input_embeddings()(input_ids)
            
#             # Convert graph tokens to same dtype as inputs_embeds
#             graph_tokens = graph_tokens.to(inputs_embeds.dtype)
#             print("inputs_embeds:", inputs_embeds.mean().item(), inputs_embeds.std().item())
#             print("graph_tokens:", graph_tokens.mean().item(), graph_tokens.std().item())   
#             # Simple concatenation or addition approach
#             # Option 1: Prepend graph tokens
#             seq_len = inputs_embeds.shape[1]
#             graph_seq_len = graph_tokens.shape[1]
            
#             # Extend attention mask
#             extended_attention_mask = torch.cat([
#                 torch.ones(B, graph_seq_len, device=dev, dtype=attention_mask.dtype),
#                 attention_mask
#             ], dim=1)
            
#             # Combine embeddings
#             combined_embeds = torch.cat([graph_tokens, inputs_embeds], dim=1)
            
#             # Adjust labels if provided
#             if labels is not None:
#                 # Pad labels with -100 for graph tokens (ignore in loss)
#                 graph_labels = torch.full((B, graph_seq_len), -100, device=dev, dtype=labels.dtype)
#                 extended_labels = torch.cat([graph_labels, labels], dim=1)
#             else:
#                 extended_labels = None
            
#             # Forward through LLaMA with modified inputs
#             outputs = self.llama(
#                 inputs_embeds=combined_embeds,
#                 attention_mask=extended_attention_mask,
#                 labels=extended_labels
#             )
#         else:
#             # Standard forward pass without graph integration
#             outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
#         return outputs  

#     def save_graph_components(self, save_dir):
#         """Save all graph-related components that are trainable"""
#         graph_components = {
#             'target_layers': self.target_layers,
#             'ast_in_proj': self.ast_in_proj.state_dict() if not isinstance(self.ast_in_proj, nn.Identity) else None,
#             'cfg_in_proj': self.cfg_in_proj.state_dict() if not isinstance(self.cfg_in_proj, nn.Identity) else None,
#             'dfg_in_proj': self.dfg_in_proj.state_dict() if not isinstance(self.dfg_in_proj, nn.Identity) else None,
#             'gnn_ast': self.gnn_ast.state_dict(),
#             'gnn_cfg': self.gnn_cfg.state_dict(),
#             'gnn_dfg': self.gnn_dfg.state_dict(),
#             'fusion_proj': self.fusion_proj.state_dict(),
#             'token_generator': self.token_generator.state_dict(),
#             'graph_integration': self.graph_integration.state_dict(),
#         }
        
#         torch.save(graph_components, os.path.join(save_dir, "graph_components_layerwise.pt"))
#         print(f"[Save] Graph components (layer-wise) saved to {save_dir}/graph_components_layerwise.pt")
    
#     def load_graph_components(self, save_dir):
#         """Load all graph-related components"""
#         graph_path = os.path.join(save_dir, "graph_components_layerwise.pt")
#         if not os.path.exists(graph_path):
#             raise FileNotFoundError(f"Graph components file not found: {graph_path}")
        
#         graph_components = torch.load(graph_path, map_location='cpu')
        
#         # Check if target layers match
#         if 'target_layers' in graph_components and graph_components['target_layers'] != self.target_layers:
#             print(f"[Load] Warning: Target layers mismatch. Saved: {graph_components['target_layers']}, Current: {self.target_layers}")
        
#         # Load each component
#         if graph_components['ast_in_proj'] is not None:
#             self.ast_in_proj.load_state_dict(graph_components['ast_in_proj'])
#         if graph_components['cfg_in_proj'] is not None:
#             self.cfg_in_proj.load_state_dict(graph_components['cfg_in_proj'])
#         if graph_components['dfg_in_proj'] is not None:
#             self.dfg_in_proj.load_state_dict(graph_components['dfg_in_proj'])
            
#         self.gnn_ast.load_state_dict(graph_components['gnn_ast'])
#         self.gnn_cfg.load_state_dict(graph_components['gnn_cfg'])
#         self.gnn_dfg.load_state_dict(graph_components['gnn_dfg'])
#         self.fusion_proj.load_state_dict(graph_components['fusion_proj'])
#         self.token_generator.load_state_dict(graph_components['token_generator'])
#         self.graph_integration.load_state_dict(graph_components['graph_integration'])
        
#         print(f"[Load] Graph components (layer-wise) loaded from {save_dir}/graph_components_layerwise.pt")

