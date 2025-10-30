#!/usr/bin/env python3
"""
Loss Plotting Script for Training Logs
Supports CSV, JSON, and TXT log files from the LossLogger
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import re

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LossPlotter:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data from various file formats"""
        if not self.log_file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file_path}")
        
        suffix = self.log_file_path.suffix.lower()
        
        if suffix == '.csv':
            self.data = self._load_csv()
        elif suffix == '.json':
            self.data = self._load_json()
        elif suffix == '.txt':
            self.data = self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        print(f"Loaded {len(self.data)} data points from {self.log_file_path}")
    
    def _load_csv(self):
        """Load from CSV file"""
        df = pd.read_csv(self.log_file_path)
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _load_json(self):
        """Load from JSON file"""
        with open(self.log_file_path, 'r') as f:
            data = json.load(f)
        
        if 'losses' in data:
            df = pd.DataFrame(data['losses'])
        else:
            df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _load_txt(self):
        """Parse TXT file (assuming format from LossLogger)"""
        data = []
        
        with open(self.log_file_path, 'r') as f:
            for line in f:
                # Skip header lines
                if 'Step' not in line or '|' not in line:
                    continue
                
                # Parse line: "Step 123 | Epoch 1 | Loss: 1.234 | EMA: 1.200 | LR: 2.00e-05 | 2024-01-01 12:00:00"
                try:
                    parts = line.strip().split(' | ')
                    step = int(re.search(r'Step\s+(\d+)', parts[0]).group(1))
                    epoch = int(re.search(r'Epoch\s+(\d+)', parts[1]).group(1))
                    loss = float(re.search(r'Loss:\s+([\d\.-]+)', parts[2]).group(1))
                    
                    entry = {'step': step, 'epoch': epoch, 'loss': loss}
                    
                    # Optional fields
                    if len(parts) > 3 and 'EMA:' in parts[3]:
                        entry['ema_loss'] = float(re.search(r'EMA:\s+([\d\.-]+)', parts[3]).group(1))
                    
                    if len(parts) > 4 and 'LR:' in parts[4]:
                        lr_match = re.search(r'LR:\s+([\d\.-e]+)', parts[4])
                        if lr_match:
                            entry['lr'] = float(lr_match.group(1))
                    
                    if len(parts) > 5:
                        entry['timestamp'] = pd.to_datetime(parts[-1].strip())
                    
                    data.append(entry)
                    
                except (AttributeError, ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue
        
        return pd.DataFrame(data)
    
    def plot_basic_loss(self, save_path=None, figsize=(12, 6)):
        """Plot basic loss curve"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot loss
        ax.plot(self.data['step'], self.data['loss'], label='Loss', alpha=0.7, linewidth=1)
        
        # Plot EMA loss if available
        if 'ema_loss' in self.data.columns and not self.data['ema_loss'].isna().all():
            ax.plot(self.data['step'], self.data['ema_loss'], label='EMA Loss', linewidth=2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set log scale if losses vary by orders of magnitude
        if self.data['loss'].max() / self.data['loss'].min() > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_detailed_analysis(self, save_path=None, figsize=(15, 10)):
        """Create a comprehensive analysis plot"""
        fig = plt.figure(figsize=figsize)
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Main loss plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data['step'], self.data['loss'], alpha=0.6, linewidth=0.8, label='Raw Loss')
        if 'ema_loss' in self.data.columns and not self.data['ema_loss'].isna().all():
            ax1.plot(self.data['step'], self.data['ema_loss'], linewidth=2, label='EMA Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss by epoch (if epoch data available)
        if 'epoch' in self.data.columns:
            ax2 = fig.add_subplot(gs[1, 0])
            epoch_losses = self.data.groupby('epoch')['loss'].mean()
            ax2.plot(epoch_losses.index, epoch_losses.values, marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Average Loss')
            ax2.set_title('Loss by Epoch')
            ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate (if available)
        if 'lr' in self.data.columns and not self.data['lr'].isna().all():
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(self.data['step'], self.data['lr'], color='orange')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. Loss distribution
        ax4 = fig.add_subplot(gs[2, 0])
        # Remove extreme outliers for better visualization
        loss_clean = self.data['loss'][self.data['loss'] < self.data['loss'].quantile(0.99)]
        ax4.hist(loss_clean, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Loss Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Loss Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Loss smoothed (rolling average)
        ax5 = fig.add_subplot(gs[2, 1])
        window_size = max(1, len(self.data) // 100)  # Adaptive window size
        rolling_loss = self.data['loss'].rolling(window=window_size, center=True).mean()
        ax5.plot(self.data['step'], rolling_loss, color='red', linewidth=2)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Loss (Smoothed)')
        ax5.set_title(f'Rolling Average Loss (window={window_size})')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Analysis - {self.log_file_path.stem}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_loss_phases(self, save_path=None, figsize=(12, 8)):
        """Identify and plot different training phases"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Calculate loss derivative to identify phases
        loss_smooth = self.data['loss'].rolling(window=max(1, len(self.data)//50), center=True).mean()
        loss_derivative = np.gradient(loss_smooth.fillna(method='bfill').fillna(method='ffill'))
        
        # Plot loss with phase coloring
        ax1.plot(self.data['step'], self.data['loss'], alpha=0.5, color='gray', linewidth=0.5)
        ax1.plot(self.data['step'], loss_smooth, linewidth=2, label='Smoothed Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss with Smoothing')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot derivative to show training dynamics
        ax2.plot(self.data['step'], loss_derivative, color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss Gradient')
        ax2.set_title('Training Dynamics (Loss Gradient)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight potential issues
        if np.any(np.isnan(self.data['loss'])) or np.any(np.isinf(self.data['loss'])):
            nan_steps = self.data['step'][np.isnan(self.data['loss']) | np.isinf(self.data['loss'])]
            for step in nan_steps:
                ax1.axvline(x=step, color='red', alpha=0.7, linestyle=':', label='NaN/Inf')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def print_statistics(self):
        """Print basic statistics about the training"""
        print("\n" + "="*50)
        print("TRAINING STATISTICS")
        print("="*50)
        
        print(f"Total training steps: {len(self.data)}")
        print(f"Final loss: {self.data['loss'].iloc[-1]:.6f}")
        print(f"Minimum loss: {self.data['loss'].min():.6f}")
        print(f"Maximum loss: {self.data['loss'].max():.6f}")
        print(f"Average loss: {self.data['loss'].mean():.6f}")
        print(f"Loss std deviation: {self.data['loss'].std():.6f}")
        
        # Check for problematic values
        nan_count = self.data['loss'].isna().sum()
        inf_count = np.isinf(self.data['loss']).sum()
        
        if nan_count > 0:
            print(f"⚠️  NaN losses found: {nan_count}")
        if inf_count > 0:
            print(f"⚠️  Infinite losses found: {inf_count}")
        
        # Training progress
        if len(self.data) > 10:
            early_loss = self.data['loss'][:10].mean()
            late_loss = self.data['loss'][-10:].mean()
            improvement = ((early_loss - late_loss) / early_loss) * 100
            print(f"Improvement from start: {improvement:.2f}%")
        
        if 'epoch' in self.data.columns:
            print(f"Epochs completed: {self.data['epoch'].max()}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Plot training losses from log files')
    parser.add_argument('log_file', help='Path to log file (CSV, JSON, or TXT)')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png', 
                       help='Output format for plots')
    parser.add_argument('--no-show', action='store_true', 
                       help='Don\'t display plots interactively')
    
    args = parser.parse_args()
    
    # Disable interactive display if requested
    if args.no_show:
        plt.ioff()
    
    try:
        # Create plotter
        plotter = LossPlotter(args.log_file)
        
        # Print statistics
        plotter.print_statistics()
        
        # Create output directory if specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(args.log_file).stem
            
            # Generate plots
            print("Generating basic loss plot...")
            plotter.plot_basic_loss(
                save_path=output_dir / f"{base_name}_basic_loss.{args.format}"
            )
            
            print("Generating detailed analysis...")
            plotter.plot_detailed_analysis(
                save_path=output_dir / f"{base_name}_detailed_analysis.{args.format}"
            )
            
            print("Generating phase analysis...")
            plotter.plot_loss_phases(
                save_path=output_dir / f"{base_name}_phase_analysis.{args.format}"
            )
            
            print(f"Plots saved to: {output_dir}")
        else:
            # Just display plots
            print("Generating basic loss plot...")
            plotter.plot_basic_loss()
            
            print("Generating detailed analysis...")
            plotter.plot_detailed_analysis()
            
            print("Generating phase analysis...")
            plotter.plot_loss_phases()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

# Convenience functions for Jupyter notebooks
def quick_plot(log_file):
    """Quick plot function for Jupyter notebooks"""
    plotter = LossPlotter(log_file)
    plotter.print_statistics()
    plotter.plot_basic_loss()
    return plotter

def analyze_training(log_file, save_plots=True):
    """Complete analysis function"""
    plotter = LossPlotter(log_file)
    plotter.print_statistics()
    
    if save_plots:
        log_path = Path(log_file)
        output_dir = log_path.parent / "plots"
        output_dir.mkdir(exist_ok=True)
        
        base_name = log_path.stem
        plotter.plot_basic_loss(output_dir / f"{base_name}_basic.png")
        plotter.plot_detailed_analysis(output_dir / f"{base_name}_detailed.png")
        plotter.plot_loss_phases(output_dir / f"{base_name}_phases.png")
        
        print(f"Plots saved to: {output_dir}")
    else:
        plotter.plot_basic_loss()
        plotter.plot_detailed_analysis()
        plotter.plot_loss_phases()
    
    return plotter

if __name__ == "__main__":
    exit(main())

# Example usage:
"""
# Command line usage:
python plot_losses.py training_logs/graph_llama_training_losses.csv --output ./plots

# In Python/Jupyter:
from plot_losses import quick_plot, analyze_training

# Quick plot
plotter = quick_plot("training_logs/graph_llama_training_losses.csv")

# Full analysis
plotter = analyze_training("training_logs/graph_llama_training_losses.json", save_plots=True)
"""