"""Generates comparative visualization figures between Original and Config B,
matching Paper Figure 5 (Training Curves) and Figure 9 / Table II-III (Evaluation Metrics).
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Styling setup
plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def plot_training_comparison(original_log, b_log, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if os.path.exists(original_log):
        df_orig = pd.read_csv(original_log)
        df_orig['rolling_return'] = df_orig['return'].rolling(window=1000, min_periods=100).mean()
        ax.plot(df_orig['episode'], df_orig['rolling_return'], label='Original (Baseline SAC)', color='#1f77b4', linewidth=2)
        
    if os.path.exists(b_log):
        df_b = pd.read_csv(b_log)
        df_b['rolling_return'] = df_b['return'].rolling(window=1000, min_periods=100).mean()
        ax.plot(df_b['episode'], df_b['rolling_return'], label='Config B (With $\mu_a$)', color='#ff7f0e', linewidth=2)
        
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Return (1,000-ep window)')
    ax.set_title('Training Average Return Curve (Comparison to Paper Fig. 5)')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_path = os.path.join(output_dir, 'training_return_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_eval_comparison(original_eval, b_eval, output_dir):
    tasks = ['left', 'straight', 'right']
    
    def compute_stats(csv_path):
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        stats = {}
        for t in tasks:
            sub = df[df['task'] == t]
            n = len(sub)
            if n == 0:
                stats[t] = {'success_rate': 0.0, 'collision_rate': 0.0, 'avg_return': 0.0, 'avg_speed': 0.0}
            else:
                stats[t] = {
                    'success_rate': (sub['outcome'] == 'arrival').mean() * 100.0,
                    'collision_rate': (sub['outcome'] == 'collision').mean() * 100.0,
                    'avg_return': sub['total_return'].mean(),
                    'avg_speed': sub['avg_speed'].mean()
                }
        return stats

    stats_orig = compute_stats(original_eval)
    stats_b = compute_stats(b_eval)
    
    if not stats_orig and not stats_b:
        print("No evaluation CSVs found to plot comparison.")
        return

    # Plot Success Rate & Collision Rate Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(tasks))
    width = 0.35

    # 1. Success Rate
    ax = axes[0]
    orig_vals = [stats_orig[t]['success_rate'] if stats_orig else 0 for t in tasks]
    b_vals = [stats_b[t]['success_rate'] if stats_b else 0 for t in tasks]
    ax.bar(x - width/2, orig_vals, width, label='Original', color='#1f77b4')
    ax.bar(x + width/2, b_vals, width, label='Config B', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Evaluation Success Rate by Task')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 2. Average Speed
    ax = axes[1]
    orig_speeds = [stats_orig[t]['avg_speed'] if stats_orig else 0 for t in tasks]
    b_speeds = [stats_b[t]['avg_speed'] if stats_b else 0 for t in tasks]
    ax.bar(x - width/2, orig_speeds, width, label='Original', color='#1f77b4')
    ax.bar(x + width/2, b_speeds, width, label='Config B', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_ylabel('Average Speed (m/s)')
    ax.set_title('Evaluation Average Speed by Task (Target = 8 m/s)')
    ax.set_ylim(0, 9)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'evaluation_metrics_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=str, default="../Original")
    parser.add_argument("--b_dir", type=str, default="../B")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    orig_train_log = os.path.join(args.original_dir, "logs", "training_progress.csv")
    b_train_log = os.path.join(args.b_dir, "logs", "training_progress.csv")
    plot_training_comparison(orig_train_log, b_train_log, args.output_dir)

    orig_eval_log = os.path.join(args.original_dir, "eval_output", "episode_metrics.csv")
    b_eval_log = os.path.join(args.b_dir, "eval_output", "episode_metrics.csv")
    plot_eval_comparison(orig_eval_log, b_eval_log, args.output_dir)