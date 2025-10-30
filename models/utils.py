import os
import csv
import json
from datetime import datetime
import torch

class LossLogger:
    """
    A utility class to log training losses to file at each iteration
    """
    def __init__(self, log_dir="training_logs", experiment_name=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"training_{timestamp}"
        
        self.experiment_name = experiment_name
        
        # File paths
        self.csv_path = os.path.join(log_dir, f"{experiment_name}_losses.csv")
        self.json_path = os.path.join(log_dir, f"{experiment_name}_losses.json")
        self.txt_path = os.path.join(log_dir, f"{experiment_name}_losses.txt")
        
        # Initialize files
        self._init_files()
        
        # Store losses for JSON export
        self.loss_history = []
        
        print(f"[LossLogger] Initialized. Logs will be saved to:")
        print(f"  CSV: {self.csv_path}")
        print(f"  JSON: {self.json_path}")
        print(f"  TXT: {self.txt_path}")
    
    def _init_files(self):
        """Initialize the log files with headers"""
        # Initialize CSV file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'loss', 'ema_loss', 'lr', 'timestamp'])
        
        # Initialize TXT file
        with open(self.txt_path, 'w') as f:
            f.write(f"Training Loss Log - {self.experiment_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_loss(self, step, epoch, loss, ema_loss=None, lr=None, extra_info=None):
        """
        Log loss at current iteration
        
        Args:
            step (int): Current training step
            epoch (int): Current epoch
            loss (float): Current loss value
            ema_loss (float, optional): Exponential moving average loss
            lr (float, optional): Current learning rate
            extra_info (dict, optional): Additional information to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare data
        log_entry = {
            'step': step,
            'epoch': epoch,
            'loss': float(loss),
            'ema_loss': float(ema_loss) if ema_loss is not None else None,
            'lr': float(lr) if lr is not None else None,
            'timestamp': timestamp
        }
        
        # Add extra info if provided
        if extra_info:
            log_entry.update(extra_info)
        
        # Add to history
        self.loss_history.append(log_entry)
        
        # Write to CSV
        self._write_csv(log_entry)
        
        # Write to TXT (human readable)
        self._write_txt(log_entry)
        
        # Update JSON file
        self._write_json()
    
    def _write_csv(self, log_entry):
        """Append entry to CSV file"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry['step'],
                log_entry['epoch'],
                log_entry['loss'],
                log_entry.get('ema_loss', ''),
                log_entry.get('lr', ''),
                log_entry['timestamp']
            ])
    
    def _write_txt(self, log_entry):
        """Append entry to TXT file"""
        with open(self.txt_path, 'a') as f:
            line = f"Step {log_entry['step']:6d} | Epoch {log_entry['epoch']:3d} | "
            line += f"Loss: {log_entry['loss']:.6f}"
            
            if log_entry.get('ema_loss'):
                line += f" | EMA: {log_entry['ema_loss']:.6f}"
            
            if log_entry.get('lr'):
                line += f" | LR: {log_entry['lr']:.2e}"
            
            line += f" | {log_entry['timestamp']}\n"
            f.write(line)
    
    def _write_json(self):
        """Write complete history to JSON file"""
        with open(self.json_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'total_steps': len(self.loss_history),
                'losses': self.loss_history
            }, f, indent=2)
    
    def get_stats(self):
        """Get basic statistics about the logged losses"""
        if not self.loss_history:
            return "No losses logged yet."
        
        losses = [entry['loss'] for entry in self.loss_history]
        return {
            'total_steps': len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'avg_loss': sum(losses) / len(losses),
            'latest_loss': losses[-1]
        }

# Simple function version (if you prefer a single function)
def log_loss_simple(step, epoch, loss, log_file="training_losses.txt", ema_loss=None, lr=None):
    """
    Simple function to log loss to a text file
    
    Args:
        step (int): Current step
        epoch (int): Current epoch  
        loss (float): Loss value
        log_file (str): Path to log file
        ema_loss (float, optional): EMA loss
        lr (float, optional): Learning rate
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create log entry
    log_line = f"Step {step:6d} | Epoch {epoch:3d} | Loss: {loss:.6f}"
    
    if ema_loss is not None:
        log_line += f" | EMA: {ema_loss:.6f}"
    
    if lr is not None:
        log_line += f" | LR: {lr:.2e}"
    
    log_line += f" | {timestamp}\n"
    
    # Append to file
    with open(log_file, 'a') as f:
        f.write(log_line)


def infer_node_feature_dim(sample_graph):
    if sample_graph is None:
        return 1
    return int(sample_graph.x.shape[1])

def nan_checker_hook(module, inputs, output):
    if isinstance(output, torch.Tensor) and (torch.isnan(output).any() or torch.isinf(output).any()):
        print(f"!!! NaN/Inf detected in output of {module.__class__.__name__} !!!")

def sanity_preview(sample):
    def stats(g, name):
        if g is None:
            return f"{name}: None"
        xshape = tuple(g.x.shape) if hasattr(g, "x") and g.x is not None else None
        ecount = g.edge_index.shape[1] if hasattr(g, "edge_index") else 0
        dtype = getattr(getattr(g, "x", None), "dtype", None)
        
        # Check for NaN/Inf in graph data
        has_nan = torch.isnan(g.x).any().item() if g.x is not None else False
        has_inf = torch.isinf(g.x).any().item() if g.x is not None else False
        
        status = ""
        if has_nan:
            status += " [NaN!]"
        if has_inf:
            status += " [Inf!]"
            
        return f"{name}: nodes={xshape[0] if xshape else 0}, feat_dim={xshape[1] if xshape else 0}, edges={ecount}, dtype={dtype}{status}"

    print("[Sanity] " + stats(sample["graph_ast"], "AST"))
    print("[Sanity] " + stats(sample["graph_cfg"], "CFG"))
    print("[Sanity] " + stats(sample["graph_dfg"], "DFG"))

def latest_ckpt(dir_):
    if not os.path.isdir(dir_):
        return None
    cands = [d for d in os.listdir(dir_) if d.startswith("ckpt_step_")]
    if not cands:
        return None
    steps = []
    for c in cands:
        try:
            steps.append(int(c.split("_")[-1]))
        except:
            pass
    if not steps:
        return None
    return os.path.join(dir_, f"ckpt_step_{max(steps)}")