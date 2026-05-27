"""
models_specificlayer.py  —  GACO graph-conditioned LLaMA

Two graph-integration designs, selected at construction time via `integration_mode`:

  "prepend"   (Design A)
      The graph is encoded -> projected to the LLaMA hidden size by
      `graph_to_llama_proj` -> prepended as soft-prompt tokens onto inputs_embeds.
      The graph lives as real sequence positions, so it is cached during prefill
      and remains attendable for the whole generation for free. No decode-time
      logic needed. `graph_integration` is NOT built in this mode.

  "crossattn" (Design B)
      The graph is encoded and exposed as an external key/value memory. A gated
      cross-attention block at each `target_layers` index enhances the text
      hidden states. `graph_to_llama_proj` is NOT built in this mode.
      Two *inference* variants share the SAME trained weights, picked by the
      runtime flag `use_cross_attn_in_decode`:
          True  (B2) -> re-attend the graph at every decode step (no fade).
          False (B1) -> only enhance during prefill; generated tokens rely on
                        the graph-enriched prompt KV cache (cheaper, can fade).
      Training is identical for B1 and B2 (teacher-forced => full-sequence pass).

Only the active path's components are built, so DistributedDataParallel never
sees a trainable-but-unused parameter (the source of the "did not receive grad"
error).

NOTE: the training script imports this as `LlamaWithGraph`. Either alias it in
models/__init__.py:
    from .models_specificlayer import LlamaWithGraphLayerSpecific as LlamaWithGraph
or import the canonical name directly.
"""

import os
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GraphNorm, GATv2Conv, global_mean_pool
from transformers import AutoModelForCausalLM

GNN_HID = 256
GNN_OUT = 256                       # per-graph embedding dim (before fusion)
GRAPH_TOKEN_NUM = 128
GRAPH_HIDDEN_DIM = 768


def _f32_no_autocast():
    """Context manager that forces float32 (disables autocast) on CUDA, no-op on CPU."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# GNN encoder
# ---------------------------------------------------------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hid_dim=GNN_HID, out_dim=GNN_OUT, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hid_dim, heads=heads, dropout=0.1)
        self.norm1 = GraphNorm(hid_dim * heads)
        self.conv2 = GATv2Conv(hid_dim * heads, hid_dim, heads=1, dropout=0.1)
        self.norm2 = GraphNorm(hid_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hid_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, GraphNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, data: PyGData):
        if data is None or data.num_nodes == 0:
            return None

        x, edge_index = data.x, data.edge_index
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

        h = self.conv1(x, edge_index)
        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        h = self.norm2(h)
        h = F.relu(h)

        if data.batch is None:
            pooled = torch.mean(h, dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(h, data.batch)

        output = self.fc(pooled)
        output = torch.clamp(output, -5.0, 5.0)
        return output


# ---------------------------------------------------------------------------
# Fused graph embedding -> graph tokens
# ---------------------------------------------------------------------------
class GraphFusionTokenGenerator(nn.Module):
    """
    Given a fused per-graph embedding [B, input_dim], project to graph_hidden_dim
    and produce graph_token_num tokens per example.
    """
    def __init__(self, input_dim, graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        self.graph_token_num = graph_token_num
        self.graph_hidden_dim = graph_hidden_dim

        self.proj = nn.Linear(input_dim, graph_hidden_dim)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.constant_(self.proj.bias, 0)

        self.pos_emb = nn.Parameter(torch.randn(graph_token_num, graph_hidden_dim) * 0.1)

        self.token_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
        )
        for m in self.token_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fused_graph_emb):  # [B, input_dim]
        if fused_graph_emb is None:
            return None
        if torch.isnan(fused_graph_emb).any() or torch.isinf(fused_graph_emb).any():
            fused_graph_emb = torch.nan_to_num(fused_graph_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        B = fused_graph_emb.size(0)
        g = self.proj(fused_graph_emb)                                  # [B, D]
        tokens = g.unsqueeze(1).repeat(1, self.graph_token_num, 1)      # [B, T, D]
        pos = self.pos_emb.unsqueeze(0).repeat(B, 1, 1)                 # [B, T, D]
        tokens = tokens + pos
        tokens = tokens + 0.1 * self.token_mlp(tokens)
        tokens = torch.clamp(tokens, -5, 5)
        return tokens


# ---------------------------------------------------------------------------
# Gated cross-attention (Design B building block)
# ---------------------------------------------------------------------------
class GatedGraphCrossAttention(nn.Module):
    """Cross-attention with gated fusion to preserve the original text context."""
    def __init__(self, text_dim, graph_dim, num_heads=8):
        super().__init__()
        self.text_dim = text_dim
        self.graph_dim = graph_dim

        self.k_proj = nn.Linear(graph_dim, text_dim)
        self.v_proj = nn.Linear(graph_dim, text_dim)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=num_heads,
            kdim=text_dim, vdim=text_dim, dropout=0.1,
        )

        self.gate_net = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(text_dim, eps=1e-5)

        # Bias the gate toward keeping the original context early in training.
        for i, layer in enumerate(self.gate_net):
            if isinstance(layer, nn.Linear):
                if i == len(self.gate_net) - 2:
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.constant_(layer.bias, -0.5)
                else:
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, text_hidden, graph_tokens, attention_mask=None):
        # text_hidden: [B, seq, D] (queries); graph_tokens: [B, G, Dg] (keys/values)
        if graph_tokens is None:
            return text_hidden
        if torch.isnan(graph_tokens).any() or torch.isinf(graph_tokens).any():
            graph_tokens = torch.nan_to_num(graph_tokens, nan=0.0, posinf=1.0, neginf=-1.0)

        text_original = text_hidden

        k = self.k_proj(graph_tokens)            # [B, G, D]
        v = self.v_proj(graph_tokens)            # [B, G, D]

        q = text_hidden.transpose(0, 1).float()  # [seq, B, D]
        k = k.transpose(0, 1).float()            # [G, B, D]
        v = v.transpose(0, 1).float()            # [G, B, D]

        try:
            attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
            attn_out = attn_out.transpose(0, 1).to(text_hidden.dtype)  # [B, seq, D]

            concat_repr = torch.cat([text_original, attn_out], dim=-1)  # [B, seq, 2D]
            gate = self.gate_net(concat_repr)                          # [B, seq, D]
            fused = gate * attn_out + (1 - gate) * text_original
            return self.norm(fused)
        except RuntimeError as e:
            print(f"[GatedCrossAttention] attention error: {e}; returning original")
            return text_hidden


# ---------------------------------------------------------------------------
# Per-layer cross-attention dispatch (Design B)
# ---------------------------------------------------------------------------
class LayerSpecificGraphIntegration(nn.Module):
    """Holds one GatedGraphCrossAttention per target layer."""
    def __init__(self, text_dim, graph_dim, target_layers=[0], num_heads=8):
        super().__init__()
        self.target_layers = set(target_layers)
        self.cross_attn = nn.ModuleDict({
            str(layer): GatedGraphCrossAttention(text_dim, graph_dim, num_heads)
            for layer in target_layers
        })
        print(f"[LayerSpecific] Graph integration enabled for layers: {sorted(target_layers)}")

    def should_apply_graph_integration(self, layer_idx):
        return layer_idx in self.target_layers

    def apply_graph_integration(self, layer_idx, text_hidden, graph_tokens, attention_mask=None):
        if not self.should_apply_graph_integration(layer_idx):
            return text_hidden
        key = str(layer_idx)
        if key in self.cross_attn:
            return self.cross_attn[key](text_hidden, graph_tokens, attention_mask)
        print(f"[LayerSpecific] Warning: no cross-attention for layer {layer_idx}")
        return text_hidden


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LlamaWithGraphLayerSpecific(nn.Module):
    def __init__(self, llama_path, tokenizer,
                 gnn_in_dim_ast, gnn_in_dim_cfg, gnn_in_dim_dfg,
                 integration_mode="prepend",          # "prepend" (A) | "crossattn" (B)
                 target_layers=[0],
                 use_cross_attn_in_decode=True,        # B only: True=B2, False=B1
                 gnn_hid=GNN_HID, gnn_out=GNN_OUT,
                 graph_token_num=GRAPH_TOKEN_NUM, graph_hidden_dim=GRAPH_HIDDEN_DIM):
        super().__init__()
        assert integration_mode in ("prepend", "crossattn"), \
            f"unknown integration_mode: {integration_mode}"

        self.integration_mode = integration_mode
        self.target_layers = target_layers
        self.use_cross_attn_in_decode = use_cross_attn_in_decode
        self.graph_token_num = graph_token_num
        self.graph_hidden_dim = graph_hidden_dim

        # Caches used only by the crossattn hook (kept defined in both modes).
        self.graph_tokens_cache = None
        self.attention_mask_cache = None

        print("[Model] loading base LLaMA (may take time)...")
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path, low_cpu_mem_usage=True, device_map=None, torch_dtype=torch.float16
        )
        for p in self.llama.parameters():
            p.requires_grad = False
        llama_hidden = self.llama.config.hidden_size

        # ---- shared graph encoder (both modes) ----
        self.ast_in_proj = nn.Linear(gnn_in_dim_ast, gnn_in_dim_ast).float() if gnn_in_dim_ast > 0 else nn.Identity()
        self.cfg_in_proj = nn.Linear(gnn_in_dim_cfg, gnn_in_dim_cfg).float() if gnn_in_dim_cfg > 0 else nn.Identity()
        self.dfg_in_proj = nn.Linear(gnn_in_dim_dfg, gnn_in_dim_dfg).float() if gnn_in_dim_dfg > 0 else nn.Identity()
        for proj in [self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight, gain=0.1)
                nn.init.constant_(proj.bias, 0)

        self.gnn_ast = SimpleGNN(in_dim=gnn_in_dim_ast, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_cfg = SimpleGNN(in_dim=gnn_in_dim_cfg, hid_dim=gnn_hid, out_dim=gnn_out).float()
        self.gnn_dfg = SimpleGNN(in_dim=gnn_in_dim_dfg, hid_dim=gnn_hid, out_dim=gnn_out).float()

        self.fusion_proj = nn.Linear(gnn_out * 3, graph_hidden_dim).float()
        nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.1)
        nn.init.constant_(self.fusion_proj.bias, 0)

        self.token_generator = GraphFusionTokenGenerator(
            input_dim=graph_hidden_dim, graph_token_num=graph_token_num, graph_hidden_dim=graph_hidden_dim
        ).float()

        # ---- mode-specific head ----
        if integration_mode == "prepend":
            self.graph_to_llama_proj = nn.Linear(graph_hidden_dim, llama_hidden).float()
            nn.init.xavier_uniform_(self.graph_to_llama_proj.weight, gain=0.1)
            nn.init.constant_(self.graph_to_llama_proj.bias, 0)
            self.graph_integration = None
            print(f"[Model] mode=prepend  (graph prepended as {graph_token_num} soft-prompt tokens)")
        else:  # crossattn
            self.graph_to_llama_proj = None
            self.graph_integration = LayerSpecificGraphIntegration(
                text_dim=llama_hidden, graph_dim=graph_hidden_dim,
                target_layers=target_layers, num_heads=16,
            ).float()
            self._setup_layer_hooks()
            print(f"[Model] mode=crossattn  target_layers={target_layers}  "
                  f"decode_cross_attn={use_cross_attn_in_decode} "
                  f"({'B2 re-attend' if use_cross_attn_in_decode else 'B1 cache-only'})")

        self._ensure_graph_components_float32()

    # ---- utilities ----
    def _ensure_graph_components_float32(self):
        for c in [self.ast_in_proj, self.cfg_in_proj, self.dfg_in_proj,
                  self.gnn_ast, self.gnn_cfg, self.gnn_dfg,
                  self.fusion_proj, self.token_generator,
                  self.graph_to_llama_proj, self.graph_integration]:
            if c is not None and not isinstance(c, nn.Identity):
                c.float()

    def _setup_layer_hooks(self):
        if hasattr(self.llama, "model") and hasattr(self.llama.model, "layers"):
            layers = self.llama.model.layers
        elif hasattr(self.llama, "transformer") and hasattr(self.llama.transformer, "h"):
            layers = self.llama.transformer.h
        else:
            print("[Model] Warning: could not locate transformer layers for hooking")
            return
        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                layers[layer_idx].register_forward_hook(self._create_layer_hook(layer_idx))
                print(f"[Model] Registered cross-attention hook for layer {layer_idx}")

    def _create_layer_hook(self, layer_idx):
        def layer_hook(module, inp, output):
            # Fires only when graph tokens have been staged (training fwd / generate wrapper).
            if self.graph_tokens_cache is None:
                return output

            hidden = output[0] if isinstance(output, tuple) else output

            # seq_len == 1 => an autoregressive decode step.
            is_decode = hidden.shape[1] == 1
            if is_decode and not self.use_cross_attn_in_decode:
                return output  # B1: rely on the graph-enriched prompt KV cache

            gt = self.graph_tokens_cache
            # Handle batch expansion (e.g. beam search / num_return_sequences).
            if gt.shape[0] != hidden.shape[0] and hidden.shape[0] % gt.shape[0] == 0:
                gt = gt.repeat_interleave(hidden.shape[0] // gt.shape[0], dim=0)

            enhanced = self.graph_integration.apply_graph_integration(
                layer_idx, hidden, gt, self.attention_mask_cache
            )
            if isinstance(output, tuple):
                return (enhanced,) + output[1:]
            return enhanced
        return layer_hook

    def _prep_batch_for_gnn(self, batch: PyGData, proj: nn.Module, device: torch.device):
        if batch is None or getattr(batch, "num_nodes", 0) == 0:
            return None
        batch = batch.to(device)
        if batch.x is not None:
            batch.x = batch.x.float()
            batch.x = torch.where(
                torch.isnan(batch.x) | torch.isinf(batch.x) | (torch.abs(batch.x) > 100),
                torch.zeros_like(batch.x), batch.x,
            )
            if batch.x.numel() > 0:
                batch.x = torch.clamp(batch.x, -10.0, 10.0)
        if not isinstance(proj, nn.Identity) and batch.x is not None:
            try:
                batch.x = torch.clamp(proj(batch.x), -10.0, 10.0)
            except Exception as e:
                print(f"[PrepGNN] projection failed: {e}; using identity")
        return batch

    def _safe_gnn(self, gnn, batch, B, dev):
        if batch is None or batch.num_nodes == 0:
            return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
        try:
            out = gnn(batch)
            if out is None:
                return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)
            if out.shape[0] != B:
                if out.shape[0] < B:
                    pad = torch.zeros((B - out.shape[0], out.shape[1]), device=dev, dtype=torch.float32)
                    out = torch.cat([out, pad], dim=0)
                else:
                    out = out[:B]
            return torch.clamp(out, -10, 10)
        except Exception as e:
            print(f"[GNN] failed: {e}; using zeros")
            return torch.zeros((B, gnn.fc.out_features), device=dev, dtype=torch.float32)

    def _encode_graph(self, ast_batch, cfg_batch, dfg_batch, B, dev):
        """Shared encoder. Returns graph tokens [B, num_tokens, graph_hidden_dim] in float32."""
        with _f32_no_autocast():
            ast_batch = self._prep_batch_for_gnn(ast_batch, self.ast_in_proj, dev)
            cfg_batch = self._prep_batch_for_gnn(cfg_batch, self.cfg_in_proj, dev)
            dfg_batch = self._prep_batch_for_gnn(dfg_batch, self.dfg_in_proj, dev)

            emb_ast = self._safe_gnn(self.gnn_ast, ast_batch, B, dev)
            emb_cfg = self._safe_gnn(self.gnn_cfg, cfg_batch, B, dev)
            emb_dfg = self._safe_gnn(self.gnn_dfg, dfg_batch, B, dev)

            fused = torch.nan_to_num(torch.cat([emb_ast, emb_cfg, emb_dfg], dim=-1))
            fused = torch.clamp(self.fusion_proj(fused), -10, 10)
            graph_tokens = self.token_generator(fused)          # [B, T, graph_hidden_dim]
        return graph_tokens

    # ---- forward (training / teacher-forced) ----
    def forward(self, input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
        B, dev = input_ids.size(0), input_ids.device
        graph_tokens = self._encode_graph(ast_batch, cfg_batch, dfg_batch, B, dev)  # [B, T, 768]

        if self.integration_mode == "prepend":
            with _f32_no_autocast():
                gt = self.graph_to_llama_proj(graph_tokens)        # [B, T, llama_hidden]
            inputs_embeds = self.llama.get_input_embeddings()(input_ids)
            gt = gt.to(inputs_embeds.dtype)

            N = gt.shape[1]
            combined = torch.cat([gt, inputs_embeds], dim=1)
            ext_mask = torch.cat(
                [torch.ones(B, N, device=dev, dtype=attention_mask.dtype), attention_mask], dim=1
            )
            ext_labels = None
            if labels is not None:
                graph_labels = torch.full((B, N), -100, device=dev, dtype=labels.dtype)
                ext_labels = torch.cat([graph_labels, labels], dim=1)

            return self.llama(inputs_embeds=combined, attention_mask=ext_mask, labels=ext_labels)

        # crossattn: stage tokens so the layer hook can read them, then run normally.
        self.graph_tokens_cache = graph_tokens
        self.attention_mask_cache = attention_mask
        try:
            outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self.graph_tokens_cache = None
            self.attention_mask_cache = None
        return outputs

    # ---- generation ----
    @torch.no_grad()
    def generate(self, input_ids, attention_mask,
                 ast_batch=None, cfg_batch=None, dfg_batch=None, **gen_kwargs):
        """
        Use this instead of model.llama.generate(...) so the graph signal is wired in.
        For crossattn, B1 vs B2 is controlled by self.use_cross_attn_in_decode.
        """
        B, dev = input_ids.size(0), input_ids.device
        graph_tokens = self._encode_graph(ast_batch, cfg_batch, dfg_batch, B, dev)

        if self.integration_mode == "prepend":
            with _f32_no_autocast():
                gt = self.graph_to_llama_proj(graph_tokens)
            inputs_embeds = self.llama.get_input_embeddings()(input_ids)
            gt = gt.to(inputs_embeds.dtype)
            N = gt.shape[1]
            combined = torch.cat([gt, inputs_embeds], dim=1)
            ext_mask = torch.cat(
                [torch.ones(B, N, device=dev, dtype=attention_mask.dtype), attention_mask], dim=1
            )
            # NOTE: with inputs_embeds, generate() returns ONLY newly generated token ids.
            return self.llama.generate(inputs_embeds=combined, attention_mask=ext_mask, **gen_kwargs)

        # crossattn: keep the cache live across the whole generation; the hook's
        # seq_len check + use_cross_attn_in_decode flag selects B1/B2 per step.
        self.graph_tokens_cache = graph_tokens
        self.attention_mask_cache = attention_mask
        try:
            return self.llama.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        finally:
            self.graph_tokens_cache = None
            self.attention_mask_cache = None

    # ---- checkpointing (mode-aware) ----
    def save_graph_components(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        components = {
            "integration_mode": self.integration_mode,
            "target_layers": self.target_layers,
            "use_cross_attn_in_decode": self.use_cross_attn_in_decode,
            "ast_in_proj": self.ast_in_proj.state_dict() if not isinstance(self.ast_in_proj, nn.Identity) else None,
            "cfg_in_proj": self.cfg_in_proj.state_dict() if not isinstance(self.cfg_in_proj, nn.Identity) else None,
            "dfg_in_proj": self.dfg_in_proj.state_dict() if not isinstance(self.dfg_in_proj, nn.Identity) else None,
            "gnn_ast": self.gnn_ast.state_dict(),
            "gnn_cfg": self.gnn_cfg.state_dict(),
            "gnn_dfg": self.gnn_dfg.state_dict(),
            "fusion_proj": self.fusion_proj.state_dict(),
            "token_generator": self.token_generator.state_dict(),
        }
        if self.integration_mode == "prepend":
            components["graph_to_llama_proj"] = self.graph_to_llama_proj.state_dict()
        else:
            components["graph_integration"] = self.graph_integration.state_dict()

        torch.save(components, os.path.join(save_dir, "graph_components_layerwise.pt"))
        print(f"[Save] Graph components ({self.integration_mode}) saved to "
              f"{save_dir}/graph_components_layerwise.pt")

    def load_graph_components(self, save_dir):
        path = os.path.join(save_dir, "graph_components_layerwise.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph components file not found: {path}")
        components = torch.load(path, map_location="cpu")

        saved_mode = components.get("integration_mode", "prepend")
        if saved_mode != self.integration_mode:
            print(f"[Load] WARNING: checkpoint mode '{saved_mode}' != model mode "
                  f"'{self.integration_mode}'. The mode-specific head will NOT be loaded.")
        if components.get("target_layers") not in (None, self.target_layers):
            print(f"[Load] WARNING: target_layers mismatch. "
                  f"Saved: {components.get('target_layers')}, current: {self.target_layers}")

        # shared encoder
        if components.get("ast_in_proj") is not None and not isinstance(self.ast_in_proj, nn.Identity):
            self.ast_in_proj.load_state_dict(components["ast_in_proj"])
        if components.get("cfg_in_proj") is not None and not isinstance(self.cfg_in_proj, nn.Identity):
            self.cfg_in_proj.load_state_dict(components["cfg_in_proj"])
        if components.get("dfg_in_proj") is not None and not isinstance(self.dfg_in_proj, nn.Identity):
            self.dfg_in_proj.load_state_dict(components["dfg_in_proj"])
        self.gnn_ast.load_state_dict(components["gnn_ast"])
        self.gnn_cfg.load_state_dict(components["gnn_cfg"])
        self.gnn_dfg.load_state_dict(components["gnn_dfg"])
        self.fusion_proj.load_state_dict(components["fusion_proj"])
        self.token_generator.load_state_dict(components["token_generator"])

        # mode-specific head (only if the checkpoint matches the model's mode)
        if self.integration_mode == "prepend" and components.get("graph_to_llama_proj") is not None:
            self.graph_to_llama_proj.load_state_dict(components["graph_to_llama_proj"])
        elif self.integration_mode == "crossattn" and components.get("graph_integration") is not None:
            self.graph_integration.load_state_dict(components["graph_integration"])
        elif saved_mode == self.integration_mode:
            print(f"[Load] WARNING: expected '{self.integration_mode}' head missing from checkpoint; "
                  f"it stays randomly initialized.")

        print(f"[Load] Graph components loaded from {path}")