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
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt
import json

from torch_geometric.nn import GATv2Conv, global_mean_pool

from models import LossLogger, PTListDataset, collate_fn, list_pt_files, SimpleGNN, GraphFusionTokenGenerator, GraphCrossAttention, LlamaWithGraph, infer_node_feature_dim, nan_checker_hook, sanity_preview, latest_ckpt

# Config
MODEL_PATH = "/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct"
PROCESSED_DIR = "processed_data/training_data"

BATCH_SIZE = 12                
GRADIENT_ACCUMULATION_STEPS = 2

LR = 2e-5
EPOCHS = 1
SEED = 42
SAVE_DIR = "checkpoints_graph_lora"
MONITOR_DIR = "training_monitors"

# Model architecture constants
GNN_HID = 256
GNN_OUT = 256  
GRAPH_TOKEN_NUM = 128
GRAPH_HIDDEN_DIM = 768

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)
set_seed(SEED)

class WeightGradientMonitor:
    """
    Monitor weight and gradient statistics for specific modules
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
        """Log statistics for current step"""
        self.step = step
        all_stats = {'step': step, 'loss': loss.item() if loss is not None else None}
        
        # Monitor different module types
        unwrapped_model = model
        if hasattr(model, 'module'):  # Handle DataParallel
            unwrapped_model = model.module
        
        # Monitor GNN modules
        for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
            if hasattr(unwrapped_model, gnn_name):
                gnn_module = getattr(unwrapped_model, gnn_name)
                gnn_stats = self.log_module_stats(gnn_name, gnn_module, step)
                all_stats[gnn_name] = gnn_stats
        
        # Monitor other graph components
        for comp_name in ['fusion_proj', 'token_generator', 'cross_attn']:
            if hasattr(unwrapped_model, comp_name):
                comp_module = getattr(unwrapped_model, comp_name)
                comp_stats = self.log_module_stats(comp_name, comp_module, step)
                all_stats[comp_name] = comp_stats
        
        # Monitor LoRA parameters specifically
        lora_stats = {}
        for name, param in unwrapped_model.llama.named_parameters():
            if param.requires_grad and ('lora_' in name or 'bias' in name):
                param_stats = self.compute_stats(param.data, f'lora_{name.replace(".", "_")}_weight')
                lora_stats.update(param_stats)
                
                if param.grad is not None:
                    grad_stats = self.compute_stats(param.grad.data, f'lora_{name.replace(".", "_")}_grad')
                    lora_stats.update(grad_stats)
        
        if lora_stats:
            all_stats['lora_adapter'] = lora_stats
            for key, value in lora_stats.items():
                self.stats['lora_adapter'][key].append({'step': step, 'value': value})
        
        # Save detailed stats periodically
        if step % self.detailed_every == 0:
            self.save_detailed_stats(step)
        
        # Log summary every few steps
        if step % self.log_every == 0:
            self.log_summary(all_stats)
        
        return all_stats
    
    def log_summary(self, stats):
        """Print summary of current statistics"""
        print(f"\n=== Monitoring Summary (Step {stats['step']}) ===")
        if stats['loss'] is not None:
            print(f"Loss: {stats['loss']:.6f}")
        
        # GNN summaries
        for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
            if gnn_name in stats:
                gnn_stats = stats[gnn_name]
                # Find gradient norms
                grad_norms = [v for k, v in gnn_stats.items() if k.endswith('_grad_norm')]
                weight_norms = [v for k, v in gnn_stats.items() if k.endswith('_weight_norm')]
                
                if grad_norms and weight_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    avg_weight_norm = np.mean(weight_norms)
                    print(f"{gnn_name.upper()}: grad_norm={avg_grad_norm:.6f}, weight_norm={avg_weight_norm:.6f}")
        
        # LoRA summary
        if 'lora_adapter' in stats:
            lora_stats = stats['lora_adapter']
            lora_grad_norms = [v for k, v in lora_stats.items() if k.endswith('_grad_norm')]
            lora_weight_norms = [v for k, v in lora_stats.items() if k.endswith('_weight_norm')]
            
            if lora_grad_norms and lora_weight_norms:
                avg_lora_grad = np.mean(lora_grad_norms)
                avg_lora_weight = np.mean(lora_weight_norms)
                print(f"LoRA: grad_norm={avg_lora_grad:.6f}, weight_norm={avg_lora_weight:.6f}")
        
        print("=" * 50)
    
    def save_detailed_stats(self, step):
        """Save detailed statistics to files"""
        # Save JSON stats
        stats_file = os.path.join(self.save_dir, f"stats_step_{step}.json")
        
        # Convert stats to serializable format
        serializable_stats = {}
        for module_name, module_stats in self.stats.items():
            serializable_stats[module_name] = {}
            for stat_name, stat_values in module_stats.items():
                serializable_stats[module_name][stat_name] = stat_values[-100:]  # Keep last 100 values
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"[Monitor] Detailed stats saved to {stats_file}")
    
    def plot_training_curves(self, step):
        """Create plots of training curves"""
        if not self.stats:
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot gradient norms over time
        plt.figure(figsize=(15, 10))
        
        for i, (module_name, module_stats) in enumerate(self.stats.items(), 1):
            plt.subplot(2, 3, i)
            
            # Find gradient norm stats
            grad_norm_keys = [k for k in module_stats.keys() if k.endswith('_grad_norm')]
            
            for key in grad_norm_keys[:5]:  # Plot first 5 to avoid clutter
                if module_stats[key]:
                    steps = [s['step'] for s in module_stats[key]]
                    values = [s['value'] for s in module_stats[key]]
                    plt.plot(steps, values, label=key.replace(f'{module_name}.', ''), alpha=0.7)
            
            plt.title(f'{module_name} Gradient Norms')
            plt.xlabel('Step')
            plt.ylabel('Gradient Norm')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"gradient_norms_step_{step}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Monitor] Plots saved to {plots_dir}")
    
    def detect_anomalies(self, stats, step):
        """Detect training anomalies"""
        anomalies = []
        
        def check_stat(stat_dict, name):
            for key, value in stat_dict.items():
                if 'has_nan' in key and value:
                    anomalies.append(f"NaN detected in {name}.{key}")
                elif 'has_inf' in key and value:
                    anomalies.append(f"Inf detected in {name}.{key}")
                elif 'grad_norm' in key and value > 100:
                    anomalies.append(f"Large gradient in {name}.{key}: {value:.2f}")
                elif 'grad_weight_ratio' in key and value > 10:
                    anomalies.append(f"Large grad/weight ratio in {name}.{key}: {value:.2f}")
        
        # Check all modules
        for module_name, module_stats in stats.items():
            if module_name not in ['step', 'loss']:
                check_stat(module_stats, module_name)
        
        if anomalies:
            print(f"\n🚨 ANOMALIES DETECTED at step {step}:")
            for anomaly in anomalies:
                print(f"  - {anomaly}")
            
            # Save anomaly report
            anomaly_file = os.path.join(self.save_dir, f"anomalies_step_{step}.txt")
            with open(anomaly_file, 'w') as f:
                f.write(f"Anomalies detected at step {step}:\n")
                for anomaly in anomalies:
                    f.write(f"- {anomaly}\n")
        
        return anomalies

def save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch, 
                   ast_dim, cfg_dim, dfg_dim, save_path, logger=None):
    """
    Centralized checkpoint saving function with proper error handling
    """
    if not accelerator.is_main_process:
        return
    
    try:
        accelerator.print(f"[Checkpoint] Starting checkpoint at step {global_step}")
        
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # Get unwrapped model
        unwrapped = accelerator.unwrap_model(model)
        
        # Save LoRA adapter
        unwrapped.llama.save_pretrained(save_path)
        accelerator.print(f"[Checkpoint] LoRA adapter saved")
        
        # Save graph components
        unwrapped.save_graph_components(save_path)
        accelerator.print(f"[Checkpoint] Graph components saved")
        
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
            },
            "lora_config": {
                "r": 12,
                "lora_alpha": 16,
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

def main():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Add NCCL timeout settings
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    if accelerator.is_main_process:
        logger = LossLogger(
            log_dir="./training_logs", 
            experiment_name="graph_llama_training"
        )
        # Initialize monitor
        monitor = WeightGradientMonitor(
            save_dir=MONITOR_DIR,
            log_every=10,  # Log summary every 10 steps
            detailed_every=100  # Save detailed stats every 100 steps
        )
    else:
        logger = None
        monitor = None
    
    device = accelerator.device
    print("[Accelerator] device:", device)
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
    sanity_preview(sample)

    # model
    model = LlamaWithGraph(MODEL_PATH, tokenizer,
                           gnn_in_dim_ast=ast_dim, gnn_in_dim_cfg=cfg_dim, gnn_in_dim_dfg=dfg_dim)

    # Register hooks for debugging
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.gnn_ast.register_forward_hook(nan_checker_hook)
    unwrapped_model.gnn_cfg.register_forward_hook(nan_checker_hook)
    unwrapped_model.gnn_dfg.register_forward_hook(nan_checker_hook)
    unwrapped_model.fusion_proj.register_forward_hook(nan_checker_hook)
    unwrapped_model.token_generator.register_forward_hook(nan_checker_hook)
    unwrapped_model.cross_attn.register_forward_hook(nan_checker_hook)

    # Register modules with monitor
    if monitor is not None:
        monitor.register_module('gnn_ast', unwrapped_model.gnn_ast)
        monitor.register_module('gnn_cfg', unwrapped_model.gnn_cfg)
        monitor.register_module('gnn_dfg', unwrapped_model.gnn_dfg)
        monitor.register_module('fusion_proj', unwrapped_model.fusion_proj)
        monitor.register_module('token_generator', unwrapped_model.token_generator)
        monitor.register_module('cross_attn', unwrapped_model.cross_attn)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        use_rslora=True
    )

    model.llama = get_peft_model(model.llama, lora_config)
    for n, p in model.llama.named_parameters():
        p.requires_grad = ("lora_" in n) or ("bias" in n)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)
    
    # prepare
    model, dataloader = accelerator.prepare(model, dataloader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LR,
        weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.95)
    )
    total_steps = EPOCHS * math.ceil(len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume logic
    resume = os.environ.get("RESUME", "0") == "1"
    start_epoch = 0
    global_step = 0
    if resume:
        last = latest_ckpt(SAVE_DIR)
        if last:
            accelerator.print(f"[Resume] Loading from {last}")
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.llama = unwrapped.llama.from_pretrained(last)
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
                optimizer.zero_grad(set_to_none=True)
                
                # Move to device with error handling
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                astb = batch["ast_batch"].to(device) if batch["ast_batch"] is not None else None
                cfgb = batch["cfg_batch"].to(device) if batch["cfg_batch"] is not None else None
                dfgb = batch["dfg_batch"].to(device) if batch["dfg_batch"] is not None else None

                t0 = time.time()
                
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                ast_batch=astb, cfg_batch=cfgb, dfg_batch=dfgb,
                                labels=labels)
                # print("Logits contains NaN?", torch.isnan(outputs.logits).any())
                loss = outputs.loss
                # print("Loss:", loss.item())
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"!!! NaN loss detected at step {global_step} !!!")
                    torch.save(batch, "poison_batch.pt")
                    raise RuntimeError("NaN loss detected")

                accelerator.backward(loss)

                total_norm = 0.0
                nan_grads = False
                
                if accelerator.sync_gradients:
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if not torch.isfinite(param_norm):
                                nan_grads = True
                                p.grad.data.zero_()
                            else:
                                total_norm += param_norm.item() ** 2
                    
                    total_norm = total_norm ** 0.5
                    
                    if nan_grads:
                        print(f"NaN gradients detected at step {global_step}, zeroed out")
                    
                    # Gradient clipping
                    max_grad_norm = 1.0
                    if total_norm > max_grad_norm:
                        clip_coef = max_grad_norm / (total_norm + 1e-6)
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(clip_coef)
                
                # # MONITOR WEIGHTS AND GRADIENTS
                # if monitor is not None and accelerator.is_main_process:
                #     stats = monitor.log_step(model, global_step, loss)
                #     anomalies = monitor.detect_anomalies(stats, global_step)
                    
                #     # Create plots periodically
                #     if global_step % 500 == 0:
                #         monitor.plot_training_curves(global_step)
                
                optimizer.step()
                scheduler.step()
                t1 = time.time()

                # meters
                with torch.no_grad():
                    bsz, seqlen = input_ids.shape
                    tokens_meter.append(bsz * seqlen / max(1e-6, (t1 - t0)))
                ema_loss = loss.item() if ema_loss is None else (smooth*ema_loss + (1-smooth)*loss.item())
                global_step += 1
                
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
                            'forward_time': t1 - t0,
                            'grad_norm': total_norm
                        }
                    )

                if global_step % log_every == 0:
                    avg_tks = sum(tokens_meter)/max(1, len(tokens_meter))
                    if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix(loss=f"{loss.item():.4f}",
                                        ema_loss=f"{ema_loss:.4f}",
                                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                        toks_s=f"{avg_tks:.0f}",
                                        grad_norm=f"{total_norm:.3f}")

                # FIXED CHECKPOINTING
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
                    raise e

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
        print("Training finished. Saved to", final_save)
        
        # Final monitoring summary
        if monitor is not None:
            monitor.plot_training_curves(global_step)
            monitor.save_detailed_stats(global_step)
            print(f"[Monitor] Final monitoring data saved to {MONITOR_DIR}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()