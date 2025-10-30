import os
from pathlib import Path
from tqdm import tqdm
import math
import time
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GraphNorm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

from torch_geometric.nn import GATv2Conv, global_mean_pool

from models import LossLogger, PTListDataset, collate_fn, list_pt_files, SimpleGNN, GraphFusionTokenGenerator, GraphCrossAttention, LlamaWithGraph, infer_node_feature_dim, nan_checker_hook, sanity_preview, latest_ckpt

# Config
MODEL_PATH = "/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct"  # model path
PROCESSED_DIR = "processed_data/training_data"   # folder with .pt files produced earlier

BATCH_SIZE = 12                
GRADIENT_ACCUMULATION_STEPS = 2

LR = 2e-5
EPOCHS = 1
SEED = 42
SAVE_DIR = "checkpoints_graph_lora"
os.makedirs(SAVE_DIR, exist_ok=True)
set_seed(SEED)


# Helper to infer node feature dims 


def main():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"
    
    # Add NCCL timeout settings
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # accelerator = Accelerator(mixed_precision="fp16")
    # accelerator = Accelerator(mixed_precision="no")

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    if accelerator.is_main_process:
        logger = LossLogger(
            log_dir="./training_logs", 
            experiment_name="graph_llama_training"
        )
    else:
        logger = None
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

    # LoRA with more conservative settings
    lora_config = LoraConfig(
        r=12,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,  # Add some dropout
        use_rslora=True

    )

    model.llama = get_peft_model(model.llama, lora_config)
    for n, p in model.llama.named_parameters():
        p.requires_grad = ("lora_" in n) or ("bias" in n)

    # Smaller batch size for debugging
    # debug_batch_size = min(BATCH_SIZE, 2)
    dataloader = DataLoader(dataset, batch_size=Bcccccbrgddindnbvchkiuenhncrlfhrcghtcfgkufjuj
    TCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)  # num_workers=0 for debugging
    
    # prepare
    model, dataloader = accelerator.prepare(model, dataloader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # More conservative optimizer settings
    # optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01, eps=1e-6)
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LR,
        weight_decay=0.01,
        eps=1e-6,          # Larger epsilon for numerical stability
        betas=(0.9, 0.95)  # More conservative beta2
    )
    total_steps = EPOCHS * math.ceil(len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume logic (unchanged)
    resume = os.environ.get("RESUME", "0") == "1"
    # resume = True
    start_epoch = 0
    global_step = 0
    if resume:
        last = latest_ckpt(SAVE_DIR)
        if last:
            accelerator.print(f"[Resume] Loading from {last}")
            unwrapped = accelerator.unwrap_model(model)
            # unwrapped.llama.from_pretrained(last)
            unwrapped.llama = unwrapped.llama.from_pretrained(last)
            meta = torch.load(os.path.join(last, "meta.pt"), map_location="cpu")
            global_step = meta.get("global_step", 0)
            start_epoch = meta.get("epoch", 0)
            optimizer.load_state_dict(meta["optim"])
            scheduler.load_state_dict(meta["sched"])

    # Training loop with enhanced debugging
    log_every = 1  # Log every step initially for debugging
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
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        epoch_start = time.time()
        
        for step, batch in enumerate(pbar, start=1):
            try:
                optimizer.zero_grad(set_to_none=True)
                
                # Move to device with error handling
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Handle graph batches safely
                astb = batch["ast_batch"].to(device) if batch["ast_batch"] is not None else None
                cfgb = batch["cfg_batch"].to(device) if batch["cfg_batch"] is not None else None
                dfgb = batch["dfg_batch"].to(device) if batch["dfg_batch"] is not None else None

                t0 = time.time()
                
                # Forward pass with mixed precision but careful GNN handling
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                ast_batch=astb, cfg_batch=cfgb, dfg_batch=dfgb,
                                labels=labels)
                loss = outputs.loss
                
                # Check for NaN loss immediately
                if torch.isnan(loss):
                    print(f"!!! NaN loss detected at step {global_step} !!!")
                    # Here you can inspect the batch that caused the issue
                    print("--- Batch causing NaN ---")
                    print(f"Input ID shape: {batch['input_ids'].shape}")
                    if batch['ast_batch']:
                        print(f"AST nodes: {batch['ast_batch'].num_nodes}, edges: {batch['ast_batch'].num_edges}")
                    if batch['cfg_batch']:
                        print(f"CFG nodes: {batch['cfg_batch'].num_nodes}, edges: {batch['cfg_batch'].num_edges}")
                    if batch['dfg_batch']:
                        print(f"DFG nodes: {batch['dfg_batch'].num_nodes}, edges: {batch['dfg_batch'].num_edges}")
                    # You can save the problematic batch for later analysis
                    torch.save(batch, "poison_batch.pt")
                    raise RuntimeError("NaN loss detected")

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     for group in optimizer.param_groups:
                #         for p in group['params']:
                #             if p.grad is not None:
                #                 torch.clamp_(p.grad, -1.0, 1.0) # Clip individual grad values

                total_norm = 0.0
                nan_grads = False
                
                if accelerator.sync_gradients:
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if not torch.isfinite(param_norm):
                                nan_grads = True
                                p.grad.data.zero_()  # Zero out bad gradients
                            else:
                                total_norm += param_norm.item() ** 2
                    
                    total_norm = total_norm ** 0.5
                else:
                    # If not syncing gradients, still check for NaN
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if not torch.isfinite(param_norm):
                                nan_grads = True
                                p.grad.data.zero_()
                    
                    if nan_grads:
                        print(f"NaN gradients detected at step {global_step}, zeroed out")
                    
                    # Adaptive gradient clipping (only when syncing gradients)
                    if accelerator.sync_gradients:
                        max_grad_norm = 1.0
                        if total_norm > max_grad_norm:
                            clip_coef = max_grad_norm / (total_norm + 1e-6)
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.data.mul_(clip_coef)
                # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                t1 = time.time()

                # meters
                with torch.no_grad():
                    bsz, seqlen = input_ids.shape
                    tokens_meter.append(bsz * seqlen / max(1e-6, (t1 - t0)))
                ema_loss = loss.item() if ema_loss is None else (smooth*ema_loss + (1-smooth)*loss.item())
                global_step += 1
                
                # Only log from main process
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
                        pbar.set_postfix(loss=f"{loss.item():.4f}",
                                        ema_loss=f"{ema_loss:.4f}",
                                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                        toks_s=f"{avg_tks:.0f}")

                # FIXED CHECKPOINTING: Proper synchronization
                if global_step % ckpt_every == 0:
                    # Ensure all processes reach this point before checkpointing
                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process:
                        accelerator.print(f"[Checkpoint] Starting checkpoint at step {global_step}")
                        
                        if logger is not None:
                            logger.log_loss(
                                step=global_step,
                                epoch=epoch,
                                loss=loss.item(),
                                ema_loss=ema_loss,
                                lr=optimizer.param_groups[0]['lr'],
                                extra_info={'status': 'checkpoint_saved'}
                            )
                        
                        save_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_step}")
                        
                        try:
                            # Get unwrapped model
                            unwrapped = accelerator.unwrap_model(model)
                            
                            # Create directory
                            os.makedirs(save_path, exist_ok=True)
                        
                            print(f"[DEBUG] Starting checkpoint to {save_path}")
                            
                            # 1. Save LoRA
                            try:
                                unwrapped.llama.save_pretrained(save_path)
                                print(f"[DEBUG] LoRA saved, files: {os.listdir(save_path)}")
                            except Exception as e:
                                print(f"[DEBUG] LoRA save failed: {e}")
                            
                            # 2. Save graph components
                            try:
                                unwrapped.save_graph_components(save_path)
                                print(f"[DEBUG] Graph components saved, files: {os.listdir(save_path)}")
                            except Exception as e:
                                print(f"[DEBUG] Graph save failed: {e}")
                            
                            # 3. Save meta
                            try:
                                torch.save({
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "optim": optimizer.state_dict(),
                                    "sched": scheduler.state_dict(),
                                }, os.path.join(save_path, "meta.pt"))
                                print(f"[DEBUG] Meta saved, final files: {os.listdir(save_path)}")
                            except Exception as e:
                                print(f"[DEBUG] Meta save failed: {e}")            
      
                            # accelerator.print(f"[Checkpoint] Successfully saved {save_path}")
                            
                        except Exception as e:
                            accelerator.print(f"[Checkpoint] Failed to save checkpoint: {e}")
                            # Continue training even if checkpoint fails
                    
                    # Wait again to ensure checkpoint is complete before continuing
                    accelerator.wait_for_everyone()
                    
            except RuntimeError as e:
                if "NaN loss detected" in str(e):
                    print("Stopping training due to NaN loss.")
                    return # Exit the function
                else:
                    raise e # Re-raise other errors

        # FIXED END OF EPOCH CHECKPOINT
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
            
            try:
                unwrapped = accelerator.unwrap_model(model)
                os.makedirs(save_path, exist_ok=True)
                unwrapped.llama.save_pretrained(save_path)
                unwrapped.save_graph_components(save_path)
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "model_config": {
                        "ast_dim": ast_dim,
                        "cfg_dim": cfg_dim, 
                        "dfg_dim": dfg_dim,
                        "gnn_hid": GNN_HID,
                        "gnn_out": GNN_OUT,
                        "graph_token_num": GRAPH_TOKEN_NUM,
                        "graph_hidden_dim": GRAPH_HIDDEN_DIM,
                    }
                }, os.path.join(save_path, "meta.pt"))
                
                accelerator.print(f"[Epoch {epoch}] done in {elapsed/60:.1f} min — ckpt at {save_path}")
                
            except Exception as e:
                accelerator.print(f"[Epoch Checkpoint] Failed to save: {e}")
        
        # Wait for all processes after epoch checkpoint
        accelerator.wait_for_everyone()

    # FIXED FINAL SAVE
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        if logger is not None:
            logger.log_loss(
                step=global_step,
                epoch=EPOCHS-1,
                loss=0,
                ema_loss=ema_loss,
                lr=0,
                extra_info={'status': 'training_completed'}
            )
        
        final_save = os.path.join(SAVE_DIR, "final")
        
        try:
            unwrapped = accelerator.unwrap_model(model)
            os.makedirs(final_save, exist_ok=True)
            unwrapped.llama.save_pretrained(final_save)
            unwrapped.save_graph_components(final_save)
            torch.save({"global_step": global_step}, os.path.join(final_save, "meta.pt"))
            print("Training finished. Saved to", final_save)
            
        except Exception as e:
            print(f"Final save failed: {e}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
