import subprocess
from datetime import datetime

# Configuration
MODEL = "apple/mobilevit-small"
DATASETS = {
    'brics2025': 'Data/brain_tumor',
    'chest_xray': 'Data/chest-xray',
    'lung_cancer': 'Data/lung-cancer'
}

# Training settings
EPOCHS = 10
BATCH_SIZE = 32
NUM_PROCESSES = 2
NUM_WORKERS = 4
SEED = 42
OPTIMIZER = 'AdamW'

# Smart grid: comparing LoRA vs No-LoRA with different learning rates
EXPERIMENTS = {
    'no_lora': [
        {'lr': 1e-3, 'weight_decay': 0.01},
        {'lr': 5e-4, 'weight_decay': 0.01},
        {'lr': 1e-4, 'weight_decay': 0.01},
    ],
    'lora': [
        {'lr': 1e-3, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1},
        {'lr': 5e-4, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1},
        {'lr': 1e-4, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1},
    ]
}

def run_experiment(dataset_name, data_dir, exp_config, use_lora=False, output_base='grid_search'):
    """Run a single experiment."""
    
    # Build command
    cmd = [
        "accelerate", "launch",
        f"--num_processes={NUM_PROCESSES}",
        "Train.py",
        "--data_dir", data_dir,
        "--model_name_or_path", MODEL,
        f"--batch_size={BATCH_SIZE}",
        f"--epochs={EPOCHS}",
        f"--lr={exp_config['lr']}",
        f"--optimizer={OPTIMIZER}",
        f"--num_workers={NUM_WORKERS}",
        f"--seed={SEED}",
        "--fast_mode"
    ]
    
    # Add LoRA parameters if needed
    if use_lora:
        cmd.extend([
            "--use_lora",
            f"--lora_r={exp_config['lora_r']}",
            f"--lora_alpha={exp_config['lora_alpha']}",
            f"--lora_dropout={exp_config['lora_dropout']}"
        ])
    else:
        cmd.append(f"--weight_decay={exp_config.get('weight_decay', 0.01)}")
    
    # Create experiment name
    exp_type = 'lora' if use_lora else 'full'
    lr_str = f"{exp_config['lr']:.0e}".replace('-', '').replace('e0', 'e-')
    exp_name = f"{dataset_name}_{exp_type}_lr{lr_str}"
    
    # Add save_dir to command
    cmd.extend(["--save_dir", f"{output_base}/{exp_name}"])
    
    print("\n" + "="*80)
    print(f"Running: {exp_name}")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Type: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print(f"Learning Rate: {exp_config['lr']}")
    if use_lora:
        print(f"LoRA Config: r={exp_config['lora_r']}, alpha={exp_config['lora_alpha']}, dropout={exp_config['lora_dropout']}")
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Completed: {exp_name}")
        return True
    else:
        print(f"✗ Failed: {exp_name}")
        print(f"Error: {result.stderr}")
        return False

def main():
    """Run smart grid search comparing LoRA vs No-LoRA."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = f'cvt13_lora_vs_full_{timestamp}'
    # Note: subdirectories will be created by Train.py
    
    # Track results
    results = {
        'timestamp': timestamp,
        'model': MODEL,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'num_processes': NUM_PROCESSES,
        'experiments': []
    }
    
    total_experiments = sum(len(EXPERIMENTS['no_lora']) + len(EXPERIMENTS['lora']) for _ in DATASETS)
    current = 0
    
    print("\n" + "="*80)
    print(f"Starting Grid Search: LoRA vs No-LoRA")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Experiments per dataset: {len(EXPERIMENTS['no_lora'])} no-LoRA + {len(EXPERIMENTS['lora'])} LoRA")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_base}")
    print("="*80)
    
    # Run all experiments
    for dataset_name, data_dir in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # No-LoRA experiments
        print(f"\n--- Full Fine-tuning Experiments ---")
        for exp_config in EXPERIMENTS['no_lora']:
            current += 1
            print(f"\nProgress: {current}/{total_experiments}")
            success = run_experiment(dataset_name, data_dir, exp_config, use_lora=False, output_base=output_base)
            
            results['experiments'].append({
                'dataset': dataset_name,
                'type': 'no_lora',
                'config': exp_config,
                'success': success
            })
        
        # LoRA experiments
        print(f"\n--- LoRA Fine-tuning Experiments ---")
        for exp_config in EXPERIMENTS['lora']:
            current += 1
            print(f"\nProgress: {current}/{total_experiments}")
            success = run_experiment(dataset_name, data_dir, exp_config, use_lora=True, output_base=output_base)
            
            results['experiments'].append({
                'dataset': dataset_name,
                'type': 'lora',
                'config': exp_config,
                'success': success
            })
    
    # Print summary
    print("\n" + "="*80)
    print("Grid Search Complete!")
    print("="*80)
    
    successful = sum(1 for exp in results['experiments'] if exp['success'])
    failed = len(results['experiments']) - successful
    
    print(f"Total experiments: {len(results['experiments'])}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_base}")
    print("="*80)

if __name__ == '__main__':
    main()