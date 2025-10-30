import os
from pathlib import Path
from tqdm import tqdm
import math
import time
from collections import deque, defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GraphNorm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
import numpy as np
import matplotlib.pyplot as plt
import json

from torch_geometric.nn import GATv2Conv, global_mean_pool

# FIXED IMPORT - Make sure this matches your new model class name
from models import DebuggingLlamaWithMultiGraphIntegration, GraphSpecificTokenGenerator, LlamaWithMultiGraphIntegration, LossLogger, PTListDataset, collate_fn, list_pt_files, SimpleGNN, infer_node_feature_dim, nan_checker_hook, sanity_preview, latest_ckpt

# Config - FIXED: Define layer configurations before use
MODEL_PATH = "/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct"
PROCESSED_DIR = "processed_data/training_data"

BATCH_SIZE = 10                
GRADIENT_ACCUMULATION_STEPS = 8

LR = 2e-5
EPOCHS = 3
SEED = 42
SAVE_DIR = "checkpoints_multimodal_graph_lora"
MONITOR_DIR = "training_monitors_multimodal"

# Model architecture constants
GNN_HID = 128
GNN_OUT = 128  
GRAPH_TOKEN_NUM = 32
GRAPH_HIDDEN_DIM = 768
ast_dim = 128
cfg_dim = 128
dfg_dim = 128

# # FIXED: Define layer configurations at module level
# AST_LAYERS = [0, 2, 4]
# CFG_LAYERS = [8, 10, 12] 
# DFG_LAYERS = [16, 18, 20]

AST_LAYERS = [0, 2]      # Reduced from [0, 2, 4]
CFG_LAYERS = [8, 10]     # Reduced from [8, 10, 12] 
DFG_LAYERS = [16, 18]    # Reduced from [16, 18, 20]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)
set_seed(SEED)

def debug_model_setup():
    """
    Replace your model creation with this debugging version
    """
    # Use debugging model instead
    model = DebuggingLlamaWithMultiGraphIntegration(
        llama_path=MODEL_PATH,
        tokenizer=tokenizer,
        gnn_in_dim_ast=ast_dim,
        gnn_in_dim_cfg=cfg_dim, 
        gnn_in_dim_dfg=dfg_dim,
        ast_layers=AST_LAYERS,
        cfg_layers=CFG_LAYERS,
        dfg_layers=DFG_LAYERS,
        gnn_hid=GNN_HID, 
        gnn_out=GNN_OUT, 
        graph_token_num=GRAPH_TOKEN_NUM, 
        graph_hidden_dim=GRAPH_HIDDEN_DIM
    )
    
    # Print debug info after a few steps
    return model

class MultiModalWeightGradientMonitor:
    """
    Monitor weight and gradient statistics for multi-modal graph components
    """
    def __init__(self, save_dir, log_every=10, detailed_every=100):
        self.save_dir = save_dir
        self.log_every = log_every
        self.detailed_every = detailed_every
        self.step = 0
        
        # Store statistics
        self.stats = defaultdict(lambda: defaultdict(list))
        self.weight_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        
        # Track weight changes
        self.prev_weights = {}
        
        os.makedirs(save_dir, exist_ok=True)
    
    def register_module(self, name, module):
        """Register a module for monitoring"""
        self.prev_weights[name] = {}
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                full_name = f"{name}.{param_name}"
                self.prev_weights[name][param_name] = param.data.clone().detach()
    
    def compute_stats(self, tensor, prefix):
        """Compute comprehensive statistics for a tensor"""
        if tensor is None:
            return {}
        
        tensor_flat = tensor.flatten()
        stats = {
            f'{prefix}_mean': tensor_flat.mean().item(),
            f'{prefix}_std': tensor_flat.std().item(),
            f'{prefix}_min': tensor_flat.min().item(),
            f'{prefix}_max': tensor_flat.max().item(),
            f'{prefix}_norm': tensor.norm().item(),
            f'{prefix}_abs_mean': tensor_flat.abs().mean().item(),
            f'{prefix}_zeros_pct': (tensor_flat == 0).float().mean().item() * 100,
        }
        
        # Check for problematic values
        stats[f'{prefix}_has_nan'] = torch.isnan(tensor).any().item()
        stats[f'{prefix}_has_inf'] = torch.isinf(tensor).any().item()
        
        return stats
    
    def log_module_stats(self, name, module, step):
        """Log statistics for a specific module"""
        module_stats = {}
        
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue
                
            full_name = f"{name}.{param_name}"
            
            # Weight statistics
            weight_stats = self.compute_stats(param.data, f'{full_name}_weight')
            module_stats.update(weight_stats)
            
            # Gradient statistics
            if param.grad is not None:
                grad_stats = self.compute_stats(param.grad.data, f'{full_name}_grad')
                module_stats.update(grad_stats)
                
                # Gradient-to-weight ratio
                grad_norm = param.grad.norm().item()
                weight_norm = param.data.norm().item()
                if weight_norm > 0:
                    module_stats[f'{full_name}_grad_weight_ratio'] = grad_norm / weight_norm
            
            # Weight change statistics
            if name in self.prev_weights and param_name in self.prev_weights[name]:
                weight_change = param.data - self.prev_weights[name][param_name]
                change_stats = self.compute_stats(weight_change, f'{full_name}_change')
                module_stats.update(change_stats)
                
                # Update previous weights
                self.prev_weights[name][param_name] = param.data.clone().detach()
        
        # Store stats for this step
        for key, value in module_stats.items():
            self.stats[name][key].append({'step': step, 'value': value})
        
        return module_stats
    
    def log_step(self, model, step, loss=None):
        """Log statistics for current step with multi-modal components"""
        self.step = step
        all_stats = {'step': step, 'loss': loss.item() if loss is not None else None}
        
        # Monitor different module types
        unwrapped_model = model
        if hasattr(model, 'module'):  # Handle DataParallel
            unwrapped_model = model.module
        
        # Monitor individual GNN modules
        for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
            if hasattr(unwrapped_model, gnn_name):
                gnn_module = getattr(unwrapped_model, gnn_name)
                gnn_stats = self.log_module_stats(gnn_name, gnn_module, step)
                all_stats[gnn_name] = gnn_stats
        
        # Monitor individual token generators
        for token_gen_name in ['ast_token_generator', 'cfg_token_generator', 'dfg_token_generator']:
            if hasattr(unwrapped_model, token_gen_name):
                token_gen_module = getattr(unwrapped_model, token_gen_name)
                token_stats = self.log_module_stats(token_gen_name, token_gen_module, step)
                all_stats[token_gen_name] = token_stats
        
        # Monitor graph integration module
        if hasattr(unwrapped_model, 'graph_integration'):
            integration_stats = self.log_module_stats('graph_integration', unwrapped_model.graph_integration, step)
            all_stats['graph_integration'] = integration_stats
        
        # # Monitor LoRA parameters specifically
        # lora_stats = {}
        # for name, param in unwrapped_model.llama.named_parameters():
        #     if param.requires_grad and ('lora_' in name or 'bias' in name):
        #         param_stats = self.compute_stats(param.data, f'lora_{name.replace(".", "_")}_weight')
        #         lora_stats.update(param_stats)
                
        #         if param.grad is not None:
        #             grad_stats = self.compute_stats(param.grad.data, f'lora_{name.replace(".", "_")}_grad')
        #             lora_stats.update(grad_stats)
        
        # if lora_stats:
        #     all_stats['lora_adapter'] = lora_stats
        #     for key, value in lora_stats.items():
        #         self.stats['lora_adapter'][key].append({'step': step, 'value': value})
        
        # Save detailed stats periodically
        if step % self.detailed_every == 0:
            self.save_detailed_stats(step)
        
        # Log summary every few steps
        if step % self.log_every == 0:
            self.log_summary(all_stats)
        
        return all_stats
    
    def log_summary(self, stats):
        """Print summary of current statistics for multi-modal components"""
        print(f"\n=== Multi-Modal Monitoring Summary (Step {stats['step']}) ===")
        if stats['loss'] is not None:
            print(f"Loss: {stats['loss']:.6f}")
        
        # Individual GNN summaries
        for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
            if gnn_name in stats:
                gnn_stats = stats[gnn_name]
                # Find gradient norms
                grad_norms = [v for k, v in gnn_stats.items() if k.endswith('_grad_norm')]
                weight_norms = [v for k, v in gnn_stats.items() if k.endswith('_weight_norm')]
                
                if grad_norms and weight_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    avg_weight_norm = np.mean(weight_norms)
                    modality = gnn_name.split('_')[1].upper()  # AST, CFG, or DFG
                    print(f"{modality} GNN: grad_norm={avg_grad_norm:.6f}, weight_norm={avg_weight_norm:.6f}")
        
        # Token generator summaries
        for token_name in ['ast_token_generator', 'cfg_token_generator', 'dfg_token_generator']:
            if token_name in stats:
                token_stats = stats[token_name]
                grad_norms = [v for k, v in token_stats.items() if k.endswith('_grad_norm')]
                if grad_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    modality = token_name.split('_')[0].upper()
                    print(f"{modality} TokenGen: grad_norm={avg_grad_norm:.6f}")
        
        # Graph integration summary
        if 'graph_integration' in stats:
            integration_stats = stats['graph_integration']
            grad_norms = [v for k, v in integration_stats.items() if k.endswith('_grad_norm')]
            if grad_norms:
                avg_grad_norm = np.mean(grad_norms)
                print(f"Graph Integration: grad_norm={avg_grad_norm:.6f}")
        
        # LoRA summary
        if 'lora_adapter' in stats:
            lora_stats = stats['lora_adapter']
            lora_grad_norms = [v for k, v in lora_stats.items() if k.endswith('_grad_norm')]
            lora_weight_norms = [v for k, v in lora_stats.items() if k.endswith('_weight_norm')]
            
            if lora_grad_norms and lora_weight_norms:
                avg_lora_grad = np.mean(lora_grad_norms)
                avg_lora_weight = np.mean(lora_weight_norms)
                print(f"LoRA: grad_norm={avg_lora_grad:.6f}, weight_norm={avg_lora_weight:.6f}")
        
        print("=" * 60)
    
    def save_detailed_stats(self, step):
        """Save detailed statistics to files"""
        # Save JSON stats
        stats_file = os.path.join(self.save_dir, f"multimodal_stats_step_{step}.json")
        
        # Convert stats to serializable format
        serializable_stats = {}
        for module_name, module_stats in self.stats.items():
            serializable_stats[module_name] = {}
            for stat_name, stat_values in module_stats.items():
                serializable_stats[module_name][stat_name] = stat_values[-100:]  # Keep last 100 values
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"[MultiModal Monitor] Detailed stats saved to {stats_file}")

def setup_optimizer_with_different_lrs(model, 
                                     gnn_lr=1e-4, 
                                     token_gen_lr=5e-5, 
                                     integration_lr=3e-5,
                                     lora_lr=2e-5,
                                     weight_decay=0.01):
    """
    FIXED: Setup optimizer with different learning rates for different components
    """
    param_groups = []
    
    # Get unwrapped model if it's wrapped
    unwrapped_model = model
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    
    # GNN parameters (higher LR since they start from scratch)
    gnn_params = []
    for component_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
        if hasattr(unwrapped_model, component_name):
            component = getattr(unwrapped_model, component_name)
            gnn_params.extend(list(component.parameters()))
    
    if gnn_params:
        param_groups.append({
            'params': gnn_params,
            'lr': gnn_lr,
            'weight_decay': weight_decay
        })
        print(f"GNN parameters: {sum(p.numel() for p in gnn_params):,} params at LR {gnn_lr}")
    
    # Token generator parameters
    token_gen_params = []
    for component_name in ['ast_token_generator', 'cfg_token_generator', 'dfg_token_generator']:
        if hasattr(unwrapped_model, component_name):
            component = getattr(unwrapped_model, component_name)
            token_gen_params.extend(list(component.parameters()))
    
    if token_gen_params:
        param_groups.append({
            'params': token_gen_params,
            'lr': token_gen_lr,
            'weight_decay': weight_decay
        })
        print(f"Token generator parameters: {sum(p.numel() for p in token_gen_params):,} params at LR {token_gen_lr}")
    
    # Integration parameters (lowest LR since this affects LLaMA directly)
    integration_params = []
    if hasattr(unwrapped_model, 'graph_integration'):
        integration_params = list(unwrapped_model.graph_integration.parameters())
    
    if integration_params:
        param_groups.append({
            'params': integration_params,
            'lr': integration_lr,
            'weight_decay': weight_decay
        })
        print(f"Integration parameters: {sum(p.numel() for p in integration_params):,} params at LR {integration_lr}")
    
    # Input projection parameters
    proj_params = []
    for component_name in ['ast_in_proj', 'cfg_in_proj', 'dfg_in_proj']:
        if hasattr(unwrapped_model, component_name):
            component = getattr(unwrapped_model, component_name)
            if not isinstance(component, nn.Identity):
                proj_params.extend(list(component.parameters()))
    
    if proj_params:
        param_groups.append({
            'params': proj_params,
            'lr': token_gen_lr,
            'weight_decay': weight_decay
        })
        print(f"Projection parameters: {sum(p.numel() for p in proj_params):,} params at LR {token_gen_lr}")
    
    # LoRA parameters
    lora_params = []
    for name, param in unwrapped_model.llama.named_parameters():
        if param.requires_grad and ('lora_' in name or 'bias' in name):
            lora_params.append(param)
    
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': lora_lr,
            'weight_decay': weight_decay
        })
        print(f"LoRA parameters: {sum(p.numel() for p in lora_params):,} params at LR {lora_lr}")
    
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

def save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch, 
                   ast_dim, cfg_dim, dfg_dim, save_path, logger=None):
    """
    Centralized checkpoint saving function for multi-modal model
    """
    if not accelerator.is_main_process:
        return
    
    try:
        accelerator.print(f"[Checkpoint] Starting multi-modal checkpoint at step {global_step}")
        
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # Get unwrapped model
        unwrapped = accelerator.unwrap_model(model)
        
        # Save LoRA adapter
        unwrapped.llama.save_pretrained(save_path)
        accelerator.print(f"[Checkpoint] LoRA adapter saved")
        
        # Save graph components
        unwrapped.save_graph_components(save_path)
        accelerator.print(f"[Checkpoint] Multi-modal graph components saved")
        
        # Save complete training state and model config
        checkpoint_data = {
            "global_step": global_step,
            "epoch": epoch,
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "model_config": {
                "model_path": MODEL_PATH,
                "ast_dim": ast_dim,
                "cfg_dim": cfg_dim, 
                "dfg_dim": dfg_dim,
                "gnn_hid": GNN_HID,
                "gnn_out": GNN_OUT,
                "graph_token_num": GRAPH_TOKEN_NUM,
                "graph_hidden_dim": GRAPH_HIDDEN_DIM,
                "ast_layers": AST_LAYERS,
                "cfg_layers": CFG_LAYERS,
                "dfg_layers": DFG_LAYERS,
            },
            "lora_config": {
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.1,
                "use_rslora": True
            },
            "training_config": {
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS
            }
        }
        
        torch.save(checkpoint_data, os.path.join(save_path, "meta.pt"))
        
        # Also save a separate config file for easier loading
        config_data = {
            "model_path": MODEL_PATH,
            "ast_dim": ast_dim,
            "cfg_dim": cfg_dim,
            "dfg_dim": dfg_dim,
            "gnn_hid": GNN_HID,
            "gnn_out": GNN_OUT,
            "graph_token_num": GRAPH_TOKEN_NUM,
            "graph_hidden_dim": GRAPH_HIDDEN_DIM,
            "ast_layers": AST_LAYERS,
            "cfg_layers": CFG_LAYERS, 
            "dfg_layers": DFG_LAYERS,
        }
        torch.save(config_data, os.path.join(save_path, "model_config.pt"))
        
        accelerator.print(f"[Checkpoint] Successfully saved to {save_path}")
        
        if logger is not None:
            logger.log_loss(
                step=global_step,
                epoch=epoch,
                loss=0,
                ema_loss=0,
                lr=0,
                extra_info={'status': 'checkpoint_saved', 'checkpoint_path': save_path}
            )
            
    except Exception as e:
        accelerator.print(f"[Checkpoint] Failed to save checkpoint: {e}")
        import traceback
        accelerator.print(f"[Checkpoint] Traceback: {traceback.format_exc()}")
def create_ddp_safe_model():
    """
    Create a DDP-safe version of your model
    """
    # Original model creation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    model = LlamaWithMultiGraphIntegration(
        llama_path=MODEL_PATH,
        tokenizer=tokenizer,
        gnn_in_dim_ast=ast_dim,
        gnn_in_dim_cfg=cfg_dim, 
        gnn_in_dim_dfg=dfg_dim,
        ast_layers=AST_LAYERS,
        cfg_layers=CFG_LAYERS,
        dfg_layers=DFG_LAYERS,
        gnn_hid=GNN_HID, 
        gnn_out=GNN_OUT, 
        graph_token_num=GRAPH_TOKEN_NUM, 
        graph_hidden_dim=GRAPH_HIDDEN_DIM
    )
    
    # Patch the forward method to ensure all parameters participate
    original_forward = model.forward
    
    def ddp_safe_forward(input_ids, attention_mask, ast_batch=None, cfg_batch=None, dfg_batch=None, labels=None):
        # Call original forward
        outputs = original_forward(input_ids, attention_mask, ast_batch, cfg_batch, dfg_batch, labels)
        
        # Force all parameters to participate in gradient computation
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            param_contribution = 0.0
            
            # Add minimal contribution from all trainable parameters
            for param in model.parameters():
                if param.requires_grad:
                    param_contribution += 1e-12 * torch.sum(param * param)
            
            outputs.loss = outputs.loss + param_contribution
            
        return outputs
    
    # Replace the forward method
    model.forward = ddp_safe_forward
    
    return model
def check_model_state(model, step):
    """Add this call in your training loop"""
    if step == 50:  # After 50 steps
        if hasattr(model, 'print_debug_summary'):
            unwrapped = model
            if hasattr(model, 'module'):
                unwrapped = model.module
            unwrapped.print_debug_summary()
def fix_hook_registration_after_lora(model):
    """
    Re-register hooks after LoRA wrapping since PEFT changes the model structure
    """
    print("[DEBUG] Re-registering hooks after LoRA application...")
    
    # Clear any existing hooks first
    if hasattr(model, '_hook_handles'):
        for handle in model._hook_handles:
            handle.remove()
    model._hook_handles = []
    
    # Find the correct layer path after LoRA wrapping
    layers = None
    layer_path = None
    
    # Try different possible paths
    possible_paths = [
        ("base_model.model.layers", lambda m: m.llama.base_model.model.layers),
        ("model.layers", lambda m: m.llama.model.layers),
        ("base_model.model.model.layers", lambda m: m.llama.base_model.model.model.layers),
    ]
    
    for path_name, path_func in possible_paths:
        try:
            layers = path_func(model)
            layer_path = path_name
            print(f"[DEBUG] Found layers at: {path_name} ({len(layers)} layers)")
            break
        except AttributeError:
            continue
    
    if layers is None:
        print("[ERROR] Could not find transformer layers after LoRA!")
        return False
    
    # Re-register hooks
    all_target_layers = set(model.ast_layers + model.cfg_layers + model.dfg_layers)
    
    for layer_idx in all_target_layers:
        if layer_idx < len(layers):
            try:
                hook_handle = layers[layer_idx].register_forward_hook(
                    model._create_layer_hook(layer_idx)
                )
                model._hook_handles.append(hook_handle)
                
                graph_types = []
                if layer_idx in model.ast_layers:
                    graph_types.append("AST")
                if layer_idx in model.cfg_layers:
                    graph_types.append("CFG")
                if layer_idx in model.dfg_layers:
                    graph_types.append("DFG")
                
                print(f"[DEBUG] Re-registered hook for layer {layer_idx} ({', '.join(graph_types)})")
                
            except Exception as e:
                print(f"[ERROR] Failed to register hook for layer {layer_idx}: {e}")
                return False
    
    print(f"[DEBUG] Successfully re-registered {len(model._hook_handles)} hooks")
    return True
    
def apply_component_gradient_scaling(model, step):
    """
    Scale gradients to balance different components
    """
    if step % 10 == 0:  # Apply every 10 steps
        
        # Scale down dominant components
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'graph_integration' in name:
                    param.grad *= 0.1  # Reduce integration gradients
                elif 'ast_' in name:
                    param.grad *= 0.5  # Reduce AST dominance
                elif 'cfg_' in name or 'dfg_' in name:
                    param.grad *= 5.0  # Boost CFG/DFG gradients

def main():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"
    
    # FIXED: Better NCCL configuration for distributed training
    os.environ["NCCL_DEBUG"] = "WARN"  # Reduced verbosity
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour timeout
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P to avoid connection issues
    os.environ["NCCL_IB_DISABLE"] = "1"   # Disable InfiniBand if not available

    # FIXED: DDP configuration for models with unused parameters
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,   # MUST be True for layer-specific injection
        static_graph=False,
        broadcast_buffers=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        logger = LossLogger(
            log_dir="./training_logs", 
            experiment_name="multimodal_graph_llama_training"
        )
        monitor = MultiModalWeightGradientMonitor(
            save_dir=MONITOR_DIR,
            log_every=10,
            detailed_every=100
        )
    else:
        logger = None
        monitor = None
    
    device = accelerator.device
    print(f"[Accelerator] device: {device}, process: {accelerator.local_process_index}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pt_files = list_pt_files(PROCESSED_DIR)
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {PROCESSED_DIR}")
    dataset = PTListDataset(pt_files)

    # infer feature dims and check data quality
    sample = None
    for s in dataset.samples:
        if s.get("graph_ast") is not None:
            sample = s
            break
    if sample is None:
        raise RuntimeError("No sample with graphs found in dataset.")
        
    ast_dim = infer_node_feature_dim(sample["graph_ast"])
    cfg_dim = infer_node_feature_dim(sample["graph_cfg"])
    dfg_dim = infer_node_feature_dim(sample["graph_dfg"])
    print(f"Inferred node feature dims: AST={ast_dim}, CFG={cfg_dim}, DFG={dfg_dim}")
    
    if accelerator.is_main_process:
        sanity_preview(sample)

    # FIXED: Create multi-modal model with correct class name and layer specifications
    # model = LlamaWithMultiGraphIntegration(  # FIXED: Use correct class name
    #     llama_path=MODEL_PATH,
    #     tokenizer=tokenizer,
    #     gnn_in_dim_ast=ast_dim,
    #     gnn_in_dim_cfg=cfg_dim, 
    #     gnn_in_dim_dfg=dfg_dim,
    #     ast_layers=AST_LAYERS,
    #     cfg_layers=CFG_LAYERS,
    #     dfg_layers=DFG_LAYERS,
    #     gnn_hid=GNN_HID, 
    #     gnn_out=GNN_OUT, 
    #     graph_token_num=GRAPH_TOKEN_NUM, 
    #     graph_hidden_dim=GRAPH_HIDDEN_DIM
    # )
    model = create_ddp_safe_model()
    print(f"LLaMA model structure:")
    if hasattr(model.llama, 'model') and hasattr(model.llama.model, 'layers'):
        total_layers = len(model.llama.model.layers)
        print(f"Found {total_layers} layers in llama.model.layers")
    elif hasattr(model.llama, 'base_model'):
        if hasattr(model.llama.base_model, 'model') and hasattr(model.llama.base_model.model, 'layers'):
            total_layers = len(model.llama.base_model.model.layers)
            print(f"Found {total_layers} layers in llama.base_model.model.layers")

    print(f"Target layers - AST: {AST_LAYERS}, CFG: {CFG_LAYERS}, DFG: {DFG_LAYERS}")
    
    # Print model configuration
    if accelerator.is_main_process:
        print(f"\n=== Multi-Modal Model Configuration ===")
        print(f"AST injection layers: {AST_LAYERS}")
        print(f"CFG injection layers: {CFG_LAYERS}")
        print(f"DFG injection layers: {DFG_LAYERS}")
        print(f"Graph tokens per modality: {GRAPH_TOKEN_NUM}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 50)
    
    # Register hooks for debugging
    if accelerator.is_main_process:
        model.gnn_ast.register_forward_hook(nan_checker_hook)
        model.gnn_cfg.register_forward_hook(nan_checker_hook)
        model.gnn_dfg.register_forward_hook(nan_checker_hook)
        model.ast_token_generator.register_forward_hook(nan_checker_hook)
        model.cfg_token_generator.register_forward_hook(nan_checker_hook)
        model.dfg_token_generator.register_forward_hook(nan_checker_hook)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        use_rslora=True
    )

    model.llama = get_peft_model(model.llama, lora_config)
    success = fix_hook_registration_after_lora(model)
    if not success:
        print("[ERROR] Hook registration failed!")
        return
    # FIXED: Ensure only LoRA parameters are trainable
    for n, p in model.llama.named_parameters():
        p.requires_grad = ("lora_" in n) or ("bias" in n)

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=2,  # Reduced from 4 to avoid potential issues
        pin_memory=True, 
        drop_last=True
    )
    
    # FIXED: Setup optimizer before preparing with accelerator
    optimizer = setup_optimizer_with_different_lrs(
        model,
        gnn_lr=2e-4,
        token_gen_lr=1e-4,
        integration_lr=1e-5,
        lora_lr=2e-5
    )
    
    total_steps = EPOCHS * math.ceil(len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # FIXED: Prepare model, optimizer, scheduler, and dataloader together
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )

    # Resume logic for multi-modal model
    resume = os.environ.get("RESUME", "0") == "1"
    start_epoch = 0
    global_step = 0
    if resume:
        last = latest_ckpt(SAVE_DIR)
        if last:
            accelerator.print(f"[Resume] Loading multi-modal model from {last}")
            unwrapped = accelerator.unwrap_model(model)
            
            # Load LoRA adapter
            unwrapped.llama = unwrapped.llama.from_pretrained(last)
            
            # Load graph components
            unwrapped.load_graph_components(last)
            
            meta = torch.load(os.path.join(last, "meta.pt"), map_location="cpu")
            global_step = meta.get("global_step", 0)
            start_epoch = meta.get("epoch", 0)
            optimizer.load_state_dict(meta["optim"])
            scheduler.load_state_dict(meta["sched"])

    # Training loop
    log_every = 1
    ckpt_every = 1000
    smooth = 0.98
    ema_loss = None
    tokens_meter = deque(maxlen=50)
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        else:
            pbar = dataloader 
        
        epoch_start = time.time()
        
        for step, batch in enumerate(pbar, start=1):
            try:
                with accelerator.accumulate(model):
                    check_model_state(model, step)

                    optimizer.zero_grad(set_to_none=True)
                    
                    # Move to device with error handling
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    
                    astb = batch["ast_batch"] if batch["ast_batch"] is not None else None
                    cfgb = batch["cfg_batch"] if batch["cfg_batch"] is not None else None
                    dfgb = batch["dfg_batch"] if batch["dfg_batch"] is not None else None

                    t0 = time.time()
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ast_batch=astb, 
                        cfg_batch=cfgb, 
                        dfg_batch=dfgb,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"!!! NaN loss detected at step {global_step} !!!")
                        torch.save(batch, "poison_batch.pt")
                        raise RuntimeError("NaN loss detected")

                    accelerator.backward(loss)

                    # FIXED: Better gradient handling
                    # if accelerator.sync_gradients:
                    #     # Gradient clipping
                    #     max_grad_norm = 1.0
                    #     accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if accelerator.sync_gradients:
                        apply_component_gradient_scaling(model, global_step)
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    t1 = time.time()

                    # meters
                    with torch.no_grad():
                        bsz, seqlen = input_ids.shape
                        tokens_meter.append(bsz * seqlen / max(1e-6, (t1 - t0)))
                    ema_loss = loss.item() if ema_loss is None else (smooth*ema_loss + (1-smooth)*loss.item())
                    global_step += 1
                    
                    # Multi-modal monitoring
                    if accelerator.is_main_process and monitor is not None:
                        monitor.log_step(model, global_step, loss)
                    
                    # Logging
                    if accelerator.is_main_process and logger is not None:
                        logger.log_loss(
                            step=global_step,
                            epoch=epoch,
                            loss=loss.item(),
                            ema_loss=ema_loss,
                            lr=optimizer.param_groups[0]['lr'],
                            extra_info={
                                'tokens_per_sec': sum(tokens_meter)/max(1, len(tokens_meter)),
                                'batch_size': bsz,
                                'sequence_length': seqlen,
                                'forward_time': t1 - t0
                            }
                        )

                    if global_step % log_every == 0:
                        avg_tks = sum(tokens_meter)/max(1, len(tokens_meter))
                        if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix(
                                loss=f"{loss.item():.4f}",
                                ema_loss=f"{ema_loss:.4f}",
                                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                toks_s=f"{avg_tks:.0f}"
                            )

                    # Checkpointing
                    if global_step % ckpt_every == 0:
                        accelerator.wait_for_everyone()
                        save_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_step}")
                        save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch,
                                      ast_dim, cfg_dim, dfg_dim, save_path, logger)
                        accelerator.wait_for_everyone()
                        
            except RuntimeError as e:
                if "NaN loss detected" in str(e):
                    print("Stopping training due to NaN loss.")
                    return
                else:
                    print(f"Training error: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    # Continue training instead of crashing
                    continue

        # End of epoch checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elapsed = time.time() - epoch_start
            if logger is not None:
                logger.log_loss(
                    step=global_step,
                    epoch=epoch,
                    loss=loss.item() if 'loss' in locals() else 0,
                    ema_loss=ema_loss,
                    lr=optimizer.param_groups[0]['lr'],
                    extra_info={
                        'status': 'epoch_completed',
                        'epoch_duration_minutes': elapsed/60,
                        'steps_in_epoch': step
                    }
                )
        
        save_path = os.path.join(SAVE_DIR, f"ckpt_epoch_{epoch}")
        save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch + 1,
                       ast_dim, cfg_dim, dfg_dim, save_path, logger)
        
        accelerator.wait_for_everyone()

    # Final save
    accelerator.wait_for_everyone()
    final_save = os.path.join(SAVE_DIR, "final")
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, EPOCHS,
                   ast_dim, cfg_dim, dfg_dim, final_save, logger)
    
    if accelerator.is_main_process:
        print("Multi-modal training finished. Saved to", final_save)
        
        # Final monitoring summary
        if monitor is not None:
            monitor.save_detailed_stats(global_step)
            print(f"[MultiModal Monitor] Final monitoring data saved to {MONITOR_DIR}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()

# # Additional utility function for loading multi-modal checkpoints
# def load_multimodal_checkpoint(checkpoint_path, device='cuda'):
#     """
#     Utility function to load a multi-modal checkpoint for inference
#     """
#     # Load config
#     config_path = os.path.join(checkpoint_path, "model_config.pt")
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Model config not found at {config_path}")
    
#     config = torch.load(config_path, map_location='cpu')
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # Create model
#     model = LlamaWithMultiGraphIntegration(
#         llama_path=config["model_path"],
#         tokenizer=tokenizer,
#         gnn_in_dim_ast=config["ast_dim"],
#         gnn_in_dim_cfg=config["cfg_dim"],
#         gnn_in_dim_dfg=config["dfg_dim"],
#         ast_layers=config["ast_layers"],
#         cfg_layers=config["cfg_layers"],
#         dfg_layers=config["dfg_layers"],
#         gnn_hid=config["gnn_hid"],
#         gnn_out=config["gnn_out"],
#         graph_token_num=config["graph_token_num"],
#         graph_hidden_dim=config["graph_hidden_dim"]
#     )
    
#     # Load LoRA weights
#     model.llama = model.llama.from_pretrained(checkpoint_path)
    
#     # Load graph components
#     model.load_graph_components(checkpoint_path)
    
#     model.to(device)
#     model.eval()
    
#     print(f"Multi-modal model loaded from {checkpoint_path}")
#     return model, tokenizer, config