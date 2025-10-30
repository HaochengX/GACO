import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GraphNorm, GATv2Conv, global_mean_pool
from transformers import AutoModelForCausalLM

GNN_HID = 128
GNN_OUT = 128                       # per-graph embedding dim
GRAPH_TOKEN_NUM = 32                # Reduced since we have 3 separate injections
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

class GraphSpecificTokenGenerator(nn.Module):
    """
    Generate tokens for a specific graph type (AST/CFG/DFG)
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

    def forward(self, graph_emb):  # graph_emb: [B, input_dim]
        if graph_emb is None:
            return None
            
        # Check for NaN/Inf
        if torch.isnan(graph_emb).any() or torch.isinf(graph_emb).any():
            print("[TokenGenerator] NaN/Inf in input!")
            graph_emb = torch.nan_to_num(graph_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
        B = graph_emb.size(0)
        g = self.proj(graph_emb)  # [B, graph_hidden_dim]
        
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
        k = self.k_proj(graph_tokens) * 0.1  # [B, G, D]
        v = self.v_proj(graph_tokens) * 0.1
        
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

class MultiGraphLayerIntegration(nn.Module):
    """
    Manages separate graph integration for AST, CFG, and DFG at different layers
    """
    def __init__(self, text_dim, graph_dim, 
                 ast_layers=[0, 2, 4], 
                 cfg_layers=[8, 10, 12], 
                 dfg_layers=[16, 18, 20], 
                 num_heads=8):
        super().__init__()
        
        self.ast_layers = set(ast_layers)
        self.cfg_layers = set(cfg_layers)
        self.dfg_layers = set(dfg_layers)
        
        # Create separate cross-attention modules for each graph type
        self.ast_cross_attn = nn.ModuleDict({
            str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
            for layer in ast_layers
        })
        
        self.cfg_cross_attn = nn.ModuleDict({
            str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
            for layer in cfg_layers
        })
        
        self.dfg_cross_attn = nn.ModuleDict({
            str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
            for layer in dfg_layers
        })
        
        print(f"[MultiGraphIntegration] AST layers: {sorted(ast_layers)}")
        print(f"[MultiGraphIntegration] CFG layers: {sorted(cfg_layers)}")
        print(f"[MultiGraphIntegration] DFG layers: {sorted(dfg_layers)}")

    def should_apply_graph_integration(self, layer_idx):
        """Check if any graph integration should be applied to this layer"""
        return (layer_idx in self.ast_layers or 
                layer_idx in self.cfg_layers or 
                layer_idx in self.dfg_layers)
    
    def apply_graph_integration(self, layer_idx, text_hidden, 
                              ast_tokens=None, cfg_tokens=None, dfg_tokens=None, 
                              attention_mask=None):
        """Apply appropriate graph integration for the specified layer"""
        modified_hidden = text_hidden
        
        # Apply AST integration if this is an AST layer
        if layer_idx in self.ast_layers and ast_tokens is not None:
            layer_key = str(layer_idx)
            if layer_key in self.ast_cross_attn:
                modified_hidden = self.ast_cross_attn[layer_key](
                    modified_hidden, ast_tokens, attention_mask
                )
        
        # Apply CFG integration if this is a CFG layer
        if layer_idx in self.cfg_layers and cfg_tokens is not None:
            layer_key = str(layer_idx)
            if layer_key in self.cfg_cross_attn:
                modified_hidden = self.cfg_cross_attn[layer_key](
                    modified_hidden, cfg_tokens, attention_mask
                )
        
        # Apply DFG integration if this is a DFG layer
        if layer_idx in self.dfg_layers and dfg_tokens is not None:
            layer_key = str(layer_idx)
            if layer_key in self.dfg_cross_attn:
                modified_hidden = self.dfg_cross_attn[layer_key](
                    modified_hidden, dfg_tokens, attention_mask
                )
        
        return modified_hidden

class LlamaWithMultiGraphIntegration(nn.Module):
    def __init__(self, llama_path, tokenizer, gnn_in_dim_ast, gnn_in_dim_cfg, gnn_in_dim_dfg,
                 ast_layers=[0, 2, 4], cfg_layers=[8, 10, 12], dfg_layers=[16, 18, 20],
                 gnn_hid=GNN_HID, gnn_out=GNN_OUT, 
                 graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        print("[Model] loading base LLaMA (may take time)...")
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path, low_cpu_mem_usage=True, device_map=None, torch_dtype=torch.float16
        )
        
        # Store layer configurations
        self.ast_layers = ast_layers
        self.cfg_layers = cfg_layers
        self.dfg_layers = dfg_layers
        
        print(f"[Model] AST integration layers: {ast_layers}")
        print(f"[Model] CFG integration layers: {cfg_layers}")
        print(f"[Model] DFG integration layers: {dfg_layers}")
        
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

        # Separate GNNs for each graph type
        self.gnn_ast = SimpleGNN(in_dim=gnn_in_dim_ast, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_cfg = SimpleGNN(in_dim=gnn_in_dim_cfg, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_dfg = SimpleGNN(in_dim=gnn_in_dim_dfg, hid_dim=gnn_hid, out_dim=gnn_out).float()

        # Separate token generators for each graph type
        self.ast_token_generator = GraphSpecificTokenGenerator(
            input_dim=gnn_out, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()
        
        self.cfg_token_generator = GraphSpecificTokenGenerator(
            input_dim=gnn_out, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()
        
        self.dfg_token_generator = GraphSpecificTokenGenerator(
            input_dim=gnn_out, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()

        # Multi-graph integration module
        self.graph_integration = MultiGraphLayerIntegration(
            text_dim=self.llama.config.hidden_size, 
            graph_dim=graph_hidden_dim,
            ast_layers=ast_layers,
            cfg_layers=cfg_layers,
            dfg_layers=dfg_layers,
            num_heads=16
        ).float()
        
        # Setup layer hooks
        self._setup_layer_hooks()
        
        # Ensure all graph-related components are in float32
        self._ensure_graph_components_float32()
    
    def _ensure_graph_components_float32(self):
        """Ensure all graph processing components are in float32 to avoid dtype mismatches"""
        components_to_float32 = [
            self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj,
            self.gnn_ast, self.gnn_cfg, self.gnn_dfg,
            self.ast_token_generator, self.cfg_token_generator, self.dfg_token_generator,
            self.graph_integration
        ]
        
        for component in components_to_float32:
            if not isinstance(component, nn.Identity):
                component.float()
    
    def _setup_layer_hooks(self):
        """Setup hooks to inject graph information into specific layers"""
        # Cache for different graph tokens
        self.ast_tokens_cache = None
        self.cfg_tokens_cache = None
        self.dfg_tokens_cache = None
        self.attention_mask_cache = None
        
        # Get the transformer layers
        if hasattr(self.llama, 'model') and hasattr(self.llama.model, 'layers'):
            layers = self.llama.model.layers
        elif hasattr(self.llama, 'transformer') and hasattr(self.llama.transformer, 'h'):
            layers = self.llama.transformer.h
        else:
            print("[Model] Warning: Could not find transformer layers for hooking")
            return
        
        # Register forward hooks for all target layers
        all_target_layers = set(self.ast_layers + self.cfg_layers + self.dfg_layers)
        
        for layer_idx in all_target_layers:
            if layer_idx < len(layers):
                layers[layer_idx].register_forward_hook(
                    self._create_layer_hook(layer_idx)
                )
                graph_types = []
                if layer_idx in self.ast_layers:
                    graph_types.append("AST")
                if layer_idx in self.cfg_layers:
                    graph_types.append("CFG")
                if layer_idx in self.dfg_layers:
                    graph_types.append("DFG")
                print(f"[Model] Registered hook for layer {layer_idx} ({', '.join(graph_types)})")
    
    def _create_layer_hook(self, layer_idx):
        """Create a forward hook for a specific layer"""
        def layer_hook(module, input, output):
            # input[0] should be the hidden states
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                
                # Apply appropriate graph integration
                if self.graph_integration.should_apply_graph_integration(layer_idx):
                    enhanced_hidden = self.graph_integration.apply_graph_integration(
                        layer_idx, hidden_states, 
                        self.ast_tokens_cache, self.cfg_tokens_cache, self.dfg_tokens_cache,
                        self.attention_mask_cache
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

        # Prep GNN inputs
        ast_batch = self._prep_batch_for_gnn(ast_batch, self.ast_in_proj, dev)
        cfg_batch = self._prep_batch_for_gnn(cfg_batch, self.cfg_in_proj, dev)
        dfg_batch = self._prep_batch_for_gnn(dfg_batch, self.dfg_in_proj, dev)

        # GNN computations for each graph type separately
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

        # Process each graph type separately
        emb_ast = safe_gnn(self.gnn_ast, ast_batch, "AST")
        emb_cfg = safe_gnn(self.gnn_cfg, cfg_batch, "CFG")
        emb_dfg = safe_gnn(self.gnn_dfg, dfg_batch, "DFG")

        # Generate tokens for each graph type separately
        def safe_token_generation(token_generator, embedding, graph_name):
            if embedding is None:
                return None
            try:
                token_generator = token_generator.float().to(dev)
                tokens = token_generator(embedding)
                if tokens is not None:
                    tokens = torch.tanh(tokens) * 0.01
                return tokens
            except Exception as e:
                print(f"[{graph_name}] Token generation failed: {e}")
                return None

        ast_tokens = safe_token_generation(self.ast_token_generator, emb_ast, "AST")
        cfg_tokens = safe_token_generation(self.cfg_token_generator, emb_cfg, "CFG")
        dfg_tokens = safe_token_generation(self.dfg_token_generator, emb_dfg, "DFG")

        # Cache tokens for layer hooks
        self.ast_tokens_cache = ast_tokens
        self.cfg_tokens_cache = cfg_tokens
        self.dfg_tokens_cache = dfg_tokens
        self.attention_mask_cache = attention_mask

        # Forward through LLaMA (hooks will inject graph info at appropriate layers)
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs

    def save_graph_components(self, save_dir):
        """Save all graph-related components that are trainable"""
        graph_components = {
            'ast_layers': self.ast_layers,
            'cfg_layers': self.cfg_layers,
            'dfg_layers': self.dfg_layers,
            'ast_in_proj': self.ast_in_proj.state_dict() if not isinstance(self.ast_in_proj, nn.Identity) else None,
            'cfg_in_proj': self.cfg_in_proj.state_dict() if not isinstance(self.cfg_in_proj, nn.Identity) else None,
            'dfg_in_proj': self.dfg_in_proj.state_dict() if not isinstance(self.dfg_in_proj, nn.Identity) else None,
            'gnn_ast': self.gnn_ast.state_dict(),
            'gnn_cfg': self.gnn_cfg.state_dict(),
            'gnn_dfg': self.gnn_dfg.state_dict(),
            'ast_token_generator': self.ast_token_generator.state_dict(),
            'cfg_token_generator': self.cfg_token_generator.state_dict(),
            'dfg_token_generator': self.dfg_token_generator.state_dict(),
            'graph_integration': self.graph_integration.state_dict(),
        }
        
        torch.save(graph_components, os.path.join(save_dir, "multi_graph_components.pt"))
        print(f"[Save] Multi-graph components saved to {save_dir}/multi_graph_components.pt")
    
    def load_graph_components(self, save_dir):
        """Load all graph-related components"""
        graph_path = os.path.join(save_dir, "multi_graph_components.pt")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph components file not found: {graph_path}")
        
        graph_components = torch.load(graph_path, map_location='cpu')
        
        # Check if layer configurations match
        if 'ast_layers' in graph_components:
            if graph_components['ast_layers'] != self.ast_layers:
                print(f"[Load] Warning: AST layers mismatch. Saved: {graph_components['ast_layers']}, Current: {self.ast_layers}")
        if 'cfg_layers' in graph_components:
            if graph_components['cfg_layers'] != self.cfg_layers:
                print(f"[Load] Warning: CFG layers mismatch. Saved: {graph_components['cfg_layers']}, Current: {self.cfg_layers}")
        if 'dfg_layers' in graph_components:
            if graph_components['dfg_layers'] != self.dfg_layers:
                print(f"[Load] Warning: DFG layers mismatch. Saved: {graph_components['dfg_layers']}, Current: {self.dfg_layers}")
        
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
        self.ast_token_generator.load_state_dict(graph_components['ast_token_generator'])
        self.cfg_token_generator.load_state_dict(graph_components['cfg_token_generator'])
        self.dfg_token_generator.load_state_dict(graph_components['dfg_token_generator'])
        self.graph_integration.load_state_dict(graph_components['graph_integration'])
        
        print(f"[Load] Multi-graph components loaded from {save_dir}/multi_graph_components.pt")



class DebuggingLlamaWithMultiGraphIntegration(LlamaWithMultiGraphIntegration):
    """
    Debugging version to identify layer injection issues
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_activation_count = {}
        self.hook_registration_status = {}
        
    def _setup_layer_hooks(self):
        """Enhanced hook setup with debugging"""
        self.ast_tokens_cache = None
        self.cfg_tokens_cache = None
        self.dfg_tokens_cache = None
        self.attention_mask_cache = None
        
        # Get transformer layers and print info
        if hasattr(self.llama, 'model') and hasattr(self.llama.model, 'layers'):
            layers = self.llama.model.layers
            model_type = "llama.model.layers"
        elif hasattr(self.llama, 'base_model') and hasattr(self.llama.base_model, 'model') and hasattr(self.llama.base_model.model, 'layers'):
            layers = self.llama.base_model.model.layers
            model_type = "llama.base_model.model.layers"
        elif hasattr(self.llama, 'transformer') and hasattr(self.llama.transformer, 'h'):
            layers = self.llama.transformer.h
            model_type = "llama.transformer.h"
        else:
            print("[DEBUG] Could not find transformer layers!")
            print(f"[DEBUG] Available attributes: {dir(self.llama)}")
            if hasattr(self.llama, 'base_model'):
                print(f"[DEBUG] Base model attributes: {dir(self.llama.base_model)}")
            return
        
        total_layers = len(layers)
        print(f"[DEBUG] Found {total_layers} layers in {model_type}")
        print(f"[DEBUG] AST target layers: {self.ast_layers}")
        print(f"[DEBUG] CFG target layers: {self.cfg_layers}")
        print(f"[DEBUG] DFG target layers: {self.dfg_layers}")
        
        # Check if target layers exceed available layers
        all_target_layers = set(self.ast_layers + self.cfg_layers + self.dfg_layers)
        invalid_layers = [l for l in all_target_layers if l >= total_layers]
        if invalid_layers:
            print(f"[DEBUG] ERROR: Target layers {invalid_layers} exceed available layers (0-{total_layers-1})")
            return
        
        # Register hooks with debugging
        for layer_idx in all_target_layers:
            if layer_idx < total_layers:
                try:
                    hook = layers[layer_idx].register_forward_hook(
                        self._create_debugging_layer_hook(layer_idx)
                    )
                    self.hook_registration_status[layer_idx] = "success"
                    self.layer_activation_count[layer_idx] = 0
                    
                    graph_types = []
                    if layer_idx in self.ast_layers:
                        graph_types.append("AST")
                    if layer_idx in self.cfg_layers:
                        graph_types.append("CFG")
                    if layer_idx in self.dfg_layers:
                        graph_types.append("DFG")
                    print(f"[DEBUG] Successfully registered hook for layer {layer_idx} ({', '.join(graph_types)})")
                    
                except Exception as e:
                    self.hook_registration_status[layer_idx] = f"failed: {e}"
                    print(f"[DEBUG] Failed to register hook for layer {layer_idx}: {e}")
    
    def _create_debugging_layer_hook(self, layer_idx):
        """Create a debugging layer hook"""
        def debugging_layer_hook(module, input, output):
            # Count activations
            self.layer_activation_count[layer_idx] += 1
            
            # Log every 10 activations
            if self.layer_activation_count[layer_idx] % 10 == 1:
                print(f"[HOOK] Layer {layer_idx} activated (count: {self.layer_activation_count[layer_idx]})")
            
            # Check which graph types should be applied
            graph_types_applied = []
            
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                original_hidden = hidden_states.clone()
                
                # Apply appropriate graph integration
                if self.graph_integration.should_apply_graph_integration(layer_idx):
                    enhanced_hidden = self.graph_integration.apply_graph_integration(
                        layer_idx, hidden_states, 
                        self.ast_tokens_cache, self.cfg_tokens_cache, self.dfg_tokens_cache,
                        self.attention_mask_cache
                    )
                    
                    # Check what was applied
                    if layer_idx in self.ast_layers and self.ast_tokens_cache is not None:
                        graph_types_applied.append("AST")
                    if layer_idx in self.cfg_layers and self.cfg_tokens_cache is not None:
                        graph_types_applied.append("CFG")
                    if layer_idx in self.dfg_layers and self.dfg_tokens_cache is not None:
                        graph_types_applied.append("DFG")
                    
                    # Log what happened
                    if self.layer_activation_count[layer_idx] % 10 == 1:
                        change_norm = (enhanced_hidden - original_hidden).norm().item()
                        print(f"[HOOK] Layer {layer_idx}: Applied {graph_types_applied}, change_norm={change_norm:.6f}")
                    
                    # Return modified output
                    if isinstance(output, tuple):
                        return (enhanced_hidden,) + output[1:]
                    else:
                        return enhanced_hidden
            
            return output
        
        return debugging_layer_hook
    
    def forward(self, input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
        """Enhanced forward with debugging"""
        print(f"[DEBUG] Forward pass started")
        print(f"[DEBUG] Input shape: {input_ids.shape}")
        print(f"[DEBUG] AST batch: {ast_batch is not None}")
        print(f"[DEBUG] CFG batch: {cfg_batch is not None}")
        print(f"[DEBUG] DFG batch: {dfg_batch is not None}")
        
        # Call parent forward
        try:
            outputs = super().forward(input_ids, attention_mask, ast_batch, cfg_batch, dfg_batch, labels)
            print(f"[DEBUG] Forward pass completed successfully")
            return outputs
        except Exception as e:
            print(f"[DEBUG] Forward pass failed: {e}")
            raise
    
    def print_debug_summary(self):
        """Print debugging summary"""
        print("\n=== DEBUGGING SUMMARY ===")
        print("Hook Registration Status:")
        for layer, status in self.hook_registration_status.items():
            print(f"  Layer {layer}: {status}")
        
        print("\nLayer Activation Counts:")
        for layer in sorted(self.layer_activation_count.keys()):
            count = self.layer_activation_count[layer]
            graph_types = []
            if layer in self.ast_layers:
                graph_types.append("AST")
            if layer in self.cfg_layers:
                graph_types.append("CFG")
            if layer in self.dfg_layers:
                graph_types.append("DFG")
            print(f"  Layer {layer} ({', '.join(graph_types)}): {count} activations")
        
        print("=" * 30)
