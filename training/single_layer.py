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

from models import LlamaWithGraphLayerSpecific, LossLogger, PTListDataset, collate_fn, list_pt_files, SimpleGNN, GraphFusionTokenGenerator,  infer_node_feature_dim, nan_checker_hook, sanity_preview, latest_ckpt

# Config - UPDATED FOR NEW DATA FORMAT
MODEL_PATH = "/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct"
PROCESSED_DIR = "processed_data/training_data"

# Conservative training parameters to prevent gradient explosion
BATCH_SIZE = 8                 # Reduced from 10
GRADIENT_ACCUMULATION_STEPS = 2  # Increased to maintain effective batch size

LR = 1e-5  # Much lower than 2e-5
EPOCHS = 2
SEED = 42
SAVE_DIR = "checkpoints_graph_lora"
MONITOR_DIR = "training_monitors"

# Model architecture constants
GNN_HID = 128
GNN_OUT = 128  
GRAPH_TOKEN_NUM = 128
GRAPH_HIDDEN_DIM = 768
MAX_LEN = 512

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)
set_seed(SEED)

class EnhancedWeightGradientMonitor:
    """
    Enhanced monitor with gradient explosion detection and health scoring
    """
    def __init__(self, save_dir, log_every=10, detailed_every=100):
        self.save_dir = save_dir
        self.log_every = log_every
        self.detailed_every = detailed_every
        self.step = 0
        
        # Store statistics with health tracking
        self.stats = defaultdict(lambda: defaultdict(list))
        self.weight_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.training_health = defaultdict(list)
        
        # Track weight changes
        self.prev_weights = {}
        
        # Health monitoring
        self.gradient_explosion_count = 0
        self.nan_detected_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def register_module(self, name, module):
        """Register a module for monitoring"""
        self.prev_weights[name] = {}
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                self.prev_weights[name][param_name] = param.data.clone().detach()
    
    def compute_stats(self, tensor, prefix):
        """Compute comprehensive statistics for a tensor"""
        if tensor is None or tensor.numel() == 0:
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
        
        # Enhanced health indicators
        stats[f'{prefix}_has_nan'] = torch.isnan(tensor).any().item()
        stats[f'{prefix}_has_inf'] = torch.isinf(tensor).any().item()
        stats[f'{prefix}_is_healthy'] = not (stats[f'{prefix}_has_nan'] or stats[f'{prefix}_has_inf'])
        
        # Distribution analysis
        if tensor_flat.numel() > 1:
            q25, q75 = torch.quantile(tensor_flat, torch.tensor([0.25, 0.75], device=tensor_flat.device))
            iqr = q75 - q25
            outlier_threshold = q75 + 1.5 * iqr
            stats[f'{prefix}_outliers_pct'] = (tensor_flat > outlier_threshold).float().mean().item() * 100
        
        return stats
    
    def log_module_stats(self, name, module, step):
        """Log statistics for a specific module with enhanced analysis"""
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
                
                # Enhanced gradient analysis
                grad_norm = param.grad.norm().item()
                weight_norm = param.data.norm().item()
                
                if weight_norm > 0:
                    ratio = grad_norm / weight_norm
                    module_stats[f'{full_name}_grad_weight_ratio'] = ratio
                    
                    # Flag potentially problematic ratios
                    if ratio > 1.0:
                        module_stats[f'{full_name}_high_grad_ratio'] = True
                
                # Track gradient explosion indicators
                if grad_norm > 10.0:
                    module_stats[f'{full_name}_gradient_explosion'] = True
                    self.gradient_explosion_count += 1
            
            # Weight change analysis
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
    
    def calculate_health_score(self, all_stats):
        """Calculate overall training health score (0-100)"""
        score = 100.0
        
        # Penalize NaN/Inf values (critical)
        for module_name, module_stats in all_stats.items():
            if module_name in ['step', 'loss']:
                continue
            for key, value in module_stats.items():
                if 'has_nan' in key and value:
                    score -= 25
                elif 'has_inf' in key and value:
                    score -= 20
                elif 'gradient_explosion' in key and value:
                    score -= 15
                elif 'high_grad_ratio' in key and value:
                    score -= 5
        
        # Penalize recent anomalies
        if self.gradient_explosion_count > 0:
            score -= min(20, self.gradient_explosion_count * 2)
        
        if self.nan_detected_count > 0:
            score -= min(30, self.nan_detected_count * 5)
        
        return max(0.0, score)
    
    def log_step(self, model, step, loss=None, grad_norm=None, lr=None):
        """Enhanced step logging with health monitoring"""
        self.step = step
        all_stats = {
            'step': step, 
            'loss': loss.item() if loss is not None else None,
            'grad_norm': grad_norm,
            'learning_rate': lr
        }
        
        # Monitor different module types
        unwrapped_model = model
        if hasattr(model, 'module'):  # Handle DataParallel
            unwrapped_model = model.module
        
        # # Monitor GNN modules
        # for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
        #     if hasattr(unwrapped_model, gnn_name):
        #         gnn_module = getattr(unwrapped_model, gnn_name)
        #         gnn_stats = self.log_module_stats(gnn_name, gnn_module, step)
        #         all_stats[gnn_name] = gnn_stats
        
        # # Monitor other graph components
        # for comp_name in ['fusion_proj', 'token_generator']:
        #     if hasattr(unwrapped_model, comp_name):
        #         comp_module = getattr(unwrapped_model, comp_name)
        #         comp_stats = self.log_module_stats(comp_name, comp_module, step)
        #         all_stats[comp_name] = comp_stats
        
        # Monitor LoRA parameters specifically
        lora_stats = {}
        lora_a_norms = []
        lora_b_norms = []
        
        for name, param in unwrapped_model.llama.named_parameters():
            if param.requires_grad and ('lora_' in name or 'bias' in name):
                param_stats = self.compute_stats(param.data, f'lora_{name.replace(".", "_")}_weight')
                lora_stats.update(param_stats)
                
                if param.grad is not None:
                    grad_stats = self.compute_stats(param.grad.data, f'lora_{name.replace(".", "_")}_grad')
                    lora_stats.update(grad_stats)
                
                # Track LoRA A and B matrix norms
                if 'lora_A' in name:
                    lora_a_norms.append(param.data.norm().item())
                elif 'lora_B' in name:
                    lora_b_norms.append(param.data.norm().item())
        
        # Calculate LoRA health metrics
        if lora_a_norms and lora_b_norms:
            lora_stats['lora_a_avg_norm'] = np.mean(lora_a_norms)
            lora_stats['lora_b_avg_norm'] = np.mean(lora_b_norms)
            lora_stats['lora_norm_ratio'] = np.mean(lora_b_norms) / max(np.mean(lora_a_norms), 1e-8)
        
        if lora_stats:
            all_stats['lora_adapter'] = lora_stats
            for key, value in lora_stats.items():
                self.stats['lora_adapter'][key].append({'step': step, 'value': value})
        
        # Calculate health score
        health_score = self.calculate_health_score(all_stats)
        all_stats['health_score'] = health_score
        
        # Store training health
        self.training_health['overall'].append({
            'step': step,
            'loss': all_stats['loss'],
            'grad_norm': grad_norm,
            'health_score': health_score,
            'learning_rate': lr
        })
        
        # Save detailed stats periodically
        if step % self.detailed_every == 0:
            self.save_detailed_stats(step)
            self.plot_training_curves(step)
        
        # Log summary
        if step % self.log_every == 0:
            self.log_summary(all_stats)
        
        # Detect and handle anomalies
        anomalies = self.detect_anomalies(all_stats, step)
        if anomalies:
            self.handle_anomalies(step, anomalies)
        
        return all_stats
    
    def log_summary(self, stats):
        """Print enhanced summary with health information"""
        print(f"\n{'='*60}")
        print(f"TRAINING MONITOR SUMMARY - Step {stats['step']}")
        print(f"{'='*60}")
        
        # Overall health
        print(f"Health Score: {stats.get('health_score', 0):.1f}/100")
        if stats.get('loss') is not None:
            print(f"Loss: {stats['loss']:.6f}")
        if stats.get('grad_norm') is not None:
            print(f"Gradient Norm: {stats['grad_norm']:.6f}")
        if stats.get('learning_rate') is not None:
            print(f"Learning Rate: {stats['learning_rate']:.2e}")
        
        # GNN summaries
        print(f"\nGRAPH NEURAL NETWORKS:")
        for gnn_name in ['gnn_ast', 'gnn_cfg', 'gnn_dfg']:
            if gnn_name in stats:
                gnn_stats = stats[gnn_name]
                grad_norms = [v for k, v in gnn_stats.items() if k.endswith('_grad_norm')]
                weight_norms = [v for k, v in gnn_stats.items() if k.endswith('_weight_norm')]
                
                if grad_norms and weight_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    avg_weight_norm = np.mean(weight_norms)
                    print(f"  {gnn_name.upper()}: grad={avg_grad_norm:.6f}, weight={avg_weight_norm:.6f}")
        
        # LoRA summary
        if 'lora_adapter' in stats:
            lora_stats = stats['lora_adapter']
            lora_grad_norms = [v for k, v in lora_stats.items() if k.endswith('_grad_norm')]
            lora_weight_norms = [v for k, v in lora_stats.items() if k.endswith('_weight_norm')]
            
            print(f"\nLORA ADAPTER:")
            if lora_grad_norms and lora_weight_norms:
                avg_lora_grad = np.mean(lora_grad_norms)
                avg_lora_weight = np.mean(lora_weight_norms)
                print(f"  Grad: {avg_lora_grad:.6f}, Weight: {avg_lora_weight:.6f}")
            
            if 'lora_norm_ratio' in lora_stats:
                print(f"  B/A Norm Ratio: {lora_stats['lora_norm_ratio']:.3f}")
        
        print("="*60)
    
    def detect_anomalies(self, stats, step):
        """Enhanced anomaly detection"""
        anomalies = []
        
        # Check for critical issues
        for module_name, module_stats in stats.items():
            if module_name in ['step', 'loss', 'grad_norm', 'learning_rate', 'health_score']:
                continue
                
            for key, value in module_stats.items():
                if 'has_nan' in key and value:
                    anomalies.append(f"CRITICAL: NaN detected in {module_name}.{key}")
                    self.nan_detected_count += 1
                elif 'has_inf' in key and value:
                    anomalies.append(f"CRITICAL: Inf detected in {module_name}.{key}")
                elif 'gradient_explosion' in key and value:
                    anomalies.append(f"WARNING: Gradient explosion in {module_name}")
                elif 'grad_norm' in key and isinstance(value, (int, float)) and value > 100:
                    anomalies.append(f"WARNING: Large gradient in {module_name}.{key}: {value:.2f}")
                elif 'grad_weight_ratio' in key and isinstance(value, (int, float)) and value > 10:
                    anomalies.append(f"WARNING: Large grad/weight ratio in {module_name}.{key}: {value:.2f}")
        
        # Check gradient norm
        if stats.get('grad_norm') is not None:
            if stats['grad_norm'] > 50:
                anomalies.append(f"CRITICAL: Very large gradient norm: {stats['grad_norm']:.2f}")
            elif stats['grad_norm'] < 1e-8:
                anomalies.append(f"WARNING: Very small gradient norm: {stats['grad_norm']:.2e}")
        
        # Check loss
        if stats.get('loss') is not None:
            if stats['loss'] > 50:
                anomalies.append(f"WARNING: Very high loss: {stats['loss']:.2f}")
        
        # Check health score
        if stats.get('health_score', 100) < 50:
            anomalies.append(f"WARNING: Low health score: {stats['health_score']:.1f}/100")
        
        return anomalies
    
    def handle_anomalies(self, step, anomalies):
        """Handle detected anomalies"""
        print(f"\n🚨 TRAINING ANOMALIES DETECTED at step {step}:")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
        
        # Save anomaly report
        anomaly_file = os.path.join(self.save_dir, f"anomalies_step_{step}.json")
        with open(anomaly_file, 'w') as f:
            json.dump({
                'step': step,
                'anomalies': anomalies,
                'timestamp': time.time(),
                'gradient_explosion_count': self.gradient_explosion_count,
                'nan_detected_count': self.nan_detected_count
            }, f, indent=2)
    
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
        
        # Add training health
        serializable_stats['training_health'] = self.training_health
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2, default=str)
        
        print(f"[Monitor] Detailed stats saved to {stats_file}")
    
    def plot_training_curves(self, step):
        """Create enhanced plots of training curves"""
        if not self.training_health['overall']:
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Training health plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        health_data = self.training_health['overall']
        
        # Loss plot
        steps = [h['step'] for h in health_data if h['loss'] is not None]
        losses = [h['loss'] for h in health_data if h['loss'] is not None]
        
        if steps and losses:
            axes[0, 0].plot(steps, losses)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_yscale('log')
        
        # Gradient norm plot
        grad_norms = [h['grad_norm'] for h in health_data if h['grad_norm'] is not None]
        grad_steps = [h['step'] for h in health_data if h['grad_norm'] is not None]
        
        if grad_steps and grad_norms:
            axes[0, 1].plot(grad_steps, grad_norms)
            axes[0, 1].set_title('Gradient Norm')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
        
        # Health score plot
        health_scores = [h['health_score'] for h in health_data if h['health_score'] is not None]
        health_steps = [h['step'] for h in health_data if h['health_score'] is not None]
        
        if health_steps and health_scores:
            axes[1, 0].plot(health_steps, health_scores)
            axes[1, 0].set_title('Training Health Score')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Health Score (0-100)')
            axes[1, 0].set_ylim(0, 100)
        
        # Learning rate plot
        lrs = [h['learning_rate'] for h in health_data if h['learning_rate'] is not None]
        lr_steps = [h['step'] for h in health_data if h['learning_rate'] is not None]
        
        if lr_steps and lrs:
            axes[1, 1].plot(lr_steps, lrs)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"training_health_step_{step}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Individual gradient norm plots for modules
        self.plot_module_gradients(step, plots_dir)
        
        print(f"[Monitor] Plots saved to {plots_dir}")
    
    def plot_module_gradients(self, step, plots_dir):
        """Plot gradient norms for individual modules"""
        if not self.stats:
            return
        
        plt.figure(figsize=(15, 10))
        
        subplot_idx = 1
        for module_name, module_stats in self.stats.items():
            if subplot_idx > 6:  # Limit to 6 subplots
                break
                
            plt.subplot(2, 3, subplot_idx)
            
            # Find gradient norm stats
            grad_norm_keys = [k for k in module_stats.keys() if k.endswith('_grad_norm')]
            
            for key in grad_norm_keys[:3]:  # Plot first 3 to avoid clutter
                if module_stats[key]:
                    steps = [s['step'] for s in module_stats[key]]
                    values = [s['value'] for s in module_stats[key]]
                    plt.plot(steps, values, label=key.split('.')[-1], alpha=0.7)
            
            plt.title(f'{module_name} Gradient Norms')
            plt.xlabel('Step')
            plt.ylabel('Gradient Norm')
            plt.legend()
            plt.yscale('log')
            
            subplot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"module_gradients_step_{step}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch, 
                   ast_dim, cfg_dim, dfg_dim, save_path, logger=None):
    """
    Enhanced checkpoint saving with new data format support
    """
    if not accelerator.is_main_process:
        return
    
    try:
        accelerator.print(f"[Checkpoint] Starting checkpoint at step {global_step}")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped = accelerator.unwrap_model(model)
        
        # Save LoRA adapter
        unwrapped.llama.save_pretrained(save_path)
        accelerator.print(f"[Checkpoint] LoRA adapter saved")
        
        # Save graph components
        unwrapped.save_graph_components(save_path)
        accelerator.print(f"[Checkpoint] Graph components saved")
        
        # Enhanced checkpoint data with new format info
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
                "target_layers": [0],  # Update if you change target layers
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
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "max_length": MAX_LEN,  # Updated for new data format
                "data_format": "corrected_sequence_format"  # Mark the data format
            }
        }
        
        torch.save(checkpoint_data, os.path.join(save_path, "meta.pt"))
        torch.save(checkpoint_data["model_config"], os.path.join(save_path, "model_config.pt"))
        
        accelerator.print(f"[Checkpoint] Successfully saved to {save_path}")
        
        if logger is not None:
            logger.log_loss(
                step=global_step, epoch=epoch, loss=0, ema_loss=0, lr=0,
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
    os.environ["NCCL_TIMEOUT"] = "1800"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False  # Changed to False for better debugging
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs]
    )
    
    device = accelerator.device
    print("[Accelerator] device:", device)
    
    # Initialize enhanced monitoring for all processes
    if accelerator.is_main_process:
        logger = LossLogger(log_dir="./training_logs", experiment_name="graph_llama_corrected_training")
        monitor = EnhancedWeightGradientMonitor(save_dir=MONITOR_DIR, log_every=5, detailed_every=50)
    else:
        logger = None
        monitor = None

    # Load tokenizer with corrected model path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset - UPDATED FOR NEW DATA FORMAT
    pt_files = list_pt_files(PROCESSED_DIR)
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {PROCESSED_DIR}")
    
    # Try to load processed_training_data.pt (from corrected preprocessing)
    dataset_file = os.path.join(PROCESSED_DIR, "processed_training_data.pt")
    if os.path.exists(dataset_file):
        print(f"Loading corrected training data from {dataset_file}")
        dataset = PTListDataset([dataset_file])
    else:
        print(f"Corrected data not found, using existing files: {pt_files}")
        dataset = PTListDataset(pt_files)

    # Infer dimensions and validate NEW DATA FORMAT
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
    
    # Validate corrected data format
    print(f"Sample validation:")
    print(f"  Input IDs shape: {sample['input_ids'].shape if isinstance(sample['input_ids'], torch.Tensor) else len(sample['input_ids'])}")
    print(f"  Labels shape: {sample['labels'].shape if isinstance(sample['labels'], torch.Tensor) else len(sample['labels'])}")
    print(f"  Masked labels count: {sum(1 for x in sample['labels'] if x == -100)}")
    print(f"  Non-masked labels count: {sum(1 for x in sample['labels'] if x != -100)}")

    model = LlamaWithGraphLayerSpecific(
        llama_path=MODEL_PATH,
        tokenizer=tokenizer,
        gnn_in_dim_ast=ast_dim, 
        gnn_in_dim_cfg=cfg_dim, 
        gnn_in_dim_dfg=dfg_dim,
        target_layers=[0],
        gnn_hid=GNN_HID,
        gnn_out=GNN_OUT,
        graph_token_num=GRAPH_TOKEN_NUM,
        graph_hidden_dim=GRAPH_HIDDEN_DIM
    )

    # Register debugging hooks
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.gnn_ast.register_forward_hook(nan_checker_hook)
    unwrapped_model.gnn_cfg.register_forward_hook(nan_checker_hook)
    unwrapped_model.gnn_dfg.register_forward_hook(nan_checker_hook)
    unwrapped_model.fusion_proj.register_forward_hook(nan_checker_hook)
    unwrapped_model.token_generator.register_forward_hook(nan_checker_hook)

    # Conservative LoRA config to prevent gradient explosion
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

    # Register modules for enhanced monitoring
    if monitor is not None:
        monitor.register_module('gnn_ast', unwrapped_model.gnn_ast)
        monitor.register_module('gnn_cfg', unwrapped_model.gnn_cfg)
        monitor.register_module('gnn_dfg', unwrapped_model.gnn_dfg)
        monitor.register_module('fusion_proj', unwrapped_model.fusion_proj)
        monitor.register_module('token_generator', unwrapped_model.token_generator)

    # Create dataloader with reduced batch size for stability
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=2,  # Reduced to avoid memory issues
        pin_memory=True, 
        drop_last=True
    )
    
    # Prepare with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # Conservative optimizer settings
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Training] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LR,
        weight_decay=0.005,  # Reduced for stability
        eps=1e-6,
        betas=(0.9, 0.95)  # More conservative beta2
    )
    
    total_steps = EPOCHS * len(dataloader)
    
    # Add warmup scheduler to prevent early gradient explosion
    def lr_lambda(step):
        warmup_steps = 50  # Warmup for first 50 steps
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    # Enhanced training loop with comprehensive monitoring
    log_every = 1
    ckpt_every = 1000  # More frequent checkpoints for debugging
    smooth = 0.98
    ema_loss = None
    tokens_meter = deque(maxlen=50)
    model.train()

    print(f"[Training] Starting training for {EPOCHS} epochs")
    print(f"[Training] Total steps: {total_steps}")
    print(f"[Training] Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(start_epoch, EPOCHS):
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        else:
            pbar = dataloader 
        
        epoch_start = time.time()
        
        for step, batch in enumerate(pbar, start=1):
            try:
                # Memory optimization
                torch.cuda.empty_cache()
                
                optimizer.zero_grad(set_to_none=True)
                
                # Enhanced batch validation for new data format
                if accelerator.is_main_process and step == 1:
                    print(f"[Batch Validation] Input shape: {batch['input_ids'].shape}")
                    print(f"[Batch Validation] Labels shape: {batch['labels'].shape}")
                    print(f"[Batch Validation] Attention mask shape: {batch['attention_mask'].shape}")
                    
                    # Check for proper masking in corrected format
                    sample_labels = batch['labels'][0]
                    masked_count = (sample_labels == -100).sum().item()
                    total_count = sample_labels.shape[0]
                    print(f"[Batch Validation] Masked tokens: {masked_count}/{total_count} ({masked_count/total_count*100:.1f}%)")

                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                astb = batch["ast_batch"].to(device) if batch["ast_batch"] is not None else None
                cfgb = batch["cfg_batch"].to(device) if batch["cfg_batch"] is not None else None
                dfgb = batch["dfg_batch"].to(device) if batch["dfg_batch"] is not None else None

                # Enhanced graph batch validation
                if accelerator.is_main_process and step == 1:
                    print(f"[Graph Validation]")
                    print(f"  AST batch: {astb.x.shape if astb is not None else 'None'}")
                    print(f"  CFG batch: {cfgb.x.shape if cfgb is not None else 'None'}")
                    print(f"  DFG batch: {dfgb.x.shape if dfgb is not None else 'None'}")

                t0 = time.time()
                
                # Forward pass with enhanced error handling
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ast_batch=astb, 
                    cfg_batch=cfgb, 
                    dfg_batch=dfgb,
                    labels=labels
                )
                loss = outputs.loss

                # Enhanced loss validation
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"!!! Invalid loss detected at step {global_step}: {loss.item()} !!!")
                    torch.save(batch, f"poison_batch_step_{global_step}.pt")
                    raise RuntimeError("NaN/Inf loss detected")

                accelerator.backward(loss)

                # Enhanced gradient monitoring and clipping
                total_norm = 0.0
                nan_grads = False
                
                if accelerator.sync_gradients:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            if not torch.isfinite(param_norm):
                                nan_grads = True
                                param.grad.data.zero_()
                                print(f"NaN gradient in {name}")
                            else:
                                total_norm += param_norm.item() ** 2
                    
                    total_norm = total_norm ** 0.5
                    
                    if nan_grads:
                        print(f"NaN gradients detected at step {global_step}, zeroed out")
                    
                    # More aggressive gradient clipping for stability
                    max_grad_norm = 0.5  # Reduced from 1.0
                    if total_norm > max_grad_norm:
                        clip_coef = max_grad_norm / (total_norm + 1e-6)
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data.mul_(clip_coef)
                        
                        if accelerator.is_main_process:
                            print(f"[GradClip] {total_norm:.3f} -> {max_grad_norm:.3f}")

                optimizer.step()
                scheduler.step()
                t1 = time.time()

                # Enhanced metrics collection
                with torch.no_grad():
                    bsz, seqlen = input_ids.shape
                    tokens_per_sec = bsz * seqlen / max(1e-6, (t1 - t0))
                    tokens_meter.append(tokens_per_sec)
                
                ema_loss = loss.item() if ema_loss is None else (smooth*ema_loss + (1-smooth)*loss.item())
                global_step += 1

                # Enhanced logging
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
                            'grad_norm': total_norm,
                            'masked_tokens_pct': (labels == -100).float().mean().item() * 100,
                            'effective_tokens': (labels != -100).sum().item(),
                            'nan_gradients': nan_grads
                        }
                    )

                # # Enhanced monitoring
                # if accelerator.is_main_process and monitor is not None:
                #     monitor.log_step(
                #         model=model,
                #         step=global_step,
                #         loss=loss,
                #         grad_norm=total_norm,
                #         lr=optimizer.param_groups[0]['lr']
                #     )

                # Enhanced progress display
                if global_step % log_every == 0:
                    avg_tks = sum(tokens_meter)/max(1, len(tokens_meter))
                    if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            ema_loss=f"{ema_loss:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                            toks_s=f"{avg_tks:.0f}",
                            grad_norm=f"{total_norm:.3f}",
                            masked_pct=f"{(labels == -100).float().mean().item()*100:.1f}%"
                        )

                # Debug info for first few steps
                if step <= 3:
                    print(f"\nStep {step} Debug:")
                    print(f"Batch size: {labels.shape[0]}")
                    print(f"Text sequence length: {labels.shape[1]}")
                    print(f"Total tokens (text): {labels.numel()}")
                    print(f"Masked tokens: {(labels == -100).sum().item()}")
                    print(f"Valid tokens: {(labels != -100).sum().item()}")
                    print(f"Masking %: {(labels == -100).float().mean() * 100:.1f}%")
                    print(f"Loss: {loss.item():.6f}")
                    print(f"Gradient norm: {total_norm:.6f}")
                    print(f"Expected total length after graphs: {labels.shape[1] + GRAPH_TOKEN_NUM}")
                    print(f"Model capacity: {MAX_LEN}")

                # Enhanced checkpointing
                if global_step % ckpt_every == 0:
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_step}")
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch,
                                  ast_dim, cfg_dim, dfg_dim, save_path, logger)
                    accelerator.wait_for_everyone()
                    
            except RuntimeError as e:
                if "NaN" in str(e) or "Inf" in str(e):
                    print("Stopping training due to numerical instability.")
                    if accelerator.is_main_process and monitor is not None:
                        monitor.handle_anomalies(global_step, [str(e)])
                    return
                else:
                    print(f"Runtime error at step {global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e

        # Enhanced end-of-epoch processing
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elapsed = time.time() - epoch_start
            avg_tokens_per_sec = sum(tokens_meter)/max(1, len(tokens_meter))
            
            print(f"\n[Epoch {epoch} Complete]")
            print(f"  Duration: {elapsed/60:.1f} minutes")
            print(f"  Steps: {step}")
            print(f"  Avg tokens/sec: {avg_tokens_per_sec:.0f}")
            print(f"  Final loss: {loss.item():.6f}")
            print(f"  EMA loss: {ema_loss:.6f}")
            
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
                        'steps_in_epoch': step,
                        'avg_tokens_per_sec': avg_tokens_per_sec
                    }
                )
        
        # Save epoch checkpoint
        save_path = os.path.join(SAVE_DIR, f"ckpt_epoch_{epoch}")
        save_checkpoint(accelerator, model, optimizer, scheduler, global_step, epoch + 1,
                       ast_dim, cfg_dim, dfg_dim, save_path, logger)
        
        accelerator.wait_for_everyone()

    # Enhanced final save
    accelerator.wait_for_everyone()
    final_save = os.path.join(SAVE_DIR, "final")
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, EPOCHS,
                   ast_dim, cfg_dim, dfg_dim, final_save, logger)
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Final model saved to: {final_save}")
        print(f"Total steps completed: {global_step}")
        print(f"Final EMA loss: {ema_loss:.6f}")
        
        # Final monitoring summary
        if monitor is not None:
            print(f"Monitoring data saved to: {MONITOR_DIR}")
            monitor.plot_training_curves(global_step)
            monitor.save_detailed_stats(global_step)
            
            # Create final summary report
            summary_file = os.path.join(MONITOR_DIR, "training_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("GRAPH LLAMA TRAINING SUMMARY\n")
                f.write("="*50 + "\n")
                f.write(f"Total steps: {global_step}\n")
                f.write(f"Total epochs: {EPOCHS}\n")
                f.write(f"Final EMA loss: {ema_loss:.6f}\n")
                f.write(f"Model path: {MODEL_PATH}\n")
                f.write(f"Data format: Corrected sequence format\n")
                f.write(f"Batch size: {BATCH_SIZE}\n")
                f.write(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}\n")
                f.write(f"Learning rate: {LR}\n")
                f.write(f"Final checkpoint: {final_save}\n")
            
            print(f"Training summary saved to: {summary_file}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()