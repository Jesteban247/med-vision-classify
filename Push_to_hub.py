import os
import json
import argparse
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def load_training_config(output_dir):
    """Load training configuration from output directory."""
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {output_dir}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_test_results(output_dir):
    """Load test results if available."""
    test_results_path = os.path.join(output_dir, 'test_results.csv')
    if os.path.exists(test_results_path):
        df = pd.read_csv(test_results_path)
        return df.to_dict('records')[0]
    return None


def create_model_card(config, test_results=None):
    """Create a comprehensive model card for the model."""
    
    base_model = config['model']
    dataset_name = Path(config['data_dir']).name
    classes = config['classes']
    is_lora = config.get('use_lora', False)
    
    # Build metrics table if test results available
    metrics_section = ""
    if test_results:
        metrics_section = f"""
## Evaluation Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Loss | {test_results.get('loss', 0):.4f} |
| Accuracy | {test_results.get('avg_accuracy', 0):.4f} |
| Precision | {test_results.get('avg_precision', 0):.4f} |
| Recall | {test_results.get('avg_recall', 0):.4f} |
| F1 Score | {test_results.get('avg_f1', 0):.4f} |
| AUC | {test_results.get('avg_auc', 0):.4f} |

### Per-Class Performance

| Class | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
"""
        for cls in classes:
            metrics_section += f"| {cls} | "
            metrics_section += f"{test_results.get(f'{cls}_accuracy', 0):.4f} | "
            metrics_section += f"{test_results.get(f'{cls}_precision', 0):.4f} | "
            metrics_section += f"{test_results.get(f'{cls}_recall', 0):.4f} | "
            metrics_section += f"{test_results.get(f'{cls}_f1', 0):.4f} | "
            metrics_section += f"{test_results.get(f'{cls}_auc', 0):.4f} |\n"
    
    # Build LoRA info if applicable
    lora_section = ""
    if is_lora:
        lora_config = config.get('lora_config', {})
        lora_section = f"""
## LoRA Configuration

This model uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

- **LoRA Rank (r)**: {lora_config.get('r', 'N/A')}
- **LoRA Alpha**: {lora_config.get('alpha', 'N/A')}
- **LoRA Dropout**: {lora_config.get('dropout', 'N/A')}
- **Target Modules**: {', '.join(lora_config.get('target_modules', []))}
- **Trainable Parameters**: {config['model_parameters']['trainable']:,} ({config['model_parameters']['trainable_percentage']:.2f}%)
- **Total Parameters**: {config['model_parameters']['total']:,}
"""
    
    model_type = "LoRA Adapter" if is_lora else "Fine-tuned Model"
    
    card_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- image-classification
- vision
- medical
- {dataset_name}
{'- lora' if is_lora else '- fine-tuned'}
- transformers
- pytorch
library_name: transformers
datasets:
- {dataset_name}
metrics:
- accuracy
- f1
- precision
- recall
- auc
---

# {base_model.split('/')[-1]} - {dataset_name.replace('-', ' ').title()} ({model_type})

## Model Description

This is a **{model_type.lower()}** of [{base_model}](https://huggingface.co/{base_model}) fine-tuned on the **{dataset_name.replace('-', ' ').title()}** dataset for medical image classification.

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Dataset**: {dataset_name}
- **Task**: Multi-class Image Classification
- **Number of Classes**: {len(classes)}
- **Classes**: {', '.join(classes)}
- **Training Method**: {'LoRA (Parameter-Efficient Fine-Tuning)' if is_lora else 'Full Fine-Tuning'}

{lora_section}

## Intended Use

This model is designed for medical image classification, specifically for classifying **{dataset_name.replace('-', ' ')}** images into the following categories:

{chr(10).join(f'- **{cls}**' for cls in classes)}

### How to Use

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
processor = AutoImageProcessor.from_pretrained("YOUR_HF_USERNAME/YOUR_MODEL_NAME")
model = AutoModelForImageClassification.from_pretrained("YOUR_HF_USERNAME/YOUR_MODEL_NAME")

# Load and preprocess image
image = Image.open("path_to_your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

print(f"Predicted class: {{model.config.id2label[predicted_class_idx]}}")
```

{metrics_section}

## Training Details

### Training Hyperparameters

- **Epochs**: {config.get('epochs', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Optimizer**: {config.get('optimizer', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}
- **Seed**: {config.get('seed', 42)}

### Model Parameters

- **Total Parameters**: {config['model_parameters']['total']:,}
- **Trainable Parameters**: {config['model_parameters']['trainable']:,} ({config['model_parameters']['trainable_percentage']:.2f}%)
- **Frozen Parameters**: {config['model_parameters']['frozen']:,}

### Training Framework

- PyTorch with Hugging Face Transformers
- Accelerate for distributed training
- PEFT for LoRA implementation{'- LoRA for parameter-efficient fine-tuning' if is_lora else ''}

## Limitations and Biases

- This model is trained on a specific medical imaging dataset and may not generalize well to other datasets or imaging modalities
- Medical AI models should always be used as decision support tools, not as standalone diagnostic systems
- Performance may vary depending on image quality, acquisition parameters, and patient demographics
- Always consult with qualified healthcare professionals for medical diagnosis

## Citation

If you use this model, please cite the original base model:

```bibtex
@article{{dehghani2021cvt,
  title={{MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer}},
  author={{Mehta, Sachin and Rastegari, Mohammad}},
  journal={{arXiv preprint arXiv:2110.02178}},
  year={{2021}}
}}
```

## Model Card Authors

This model card was created as part of a medical image classification project.

## Training Details

**Training Date**: {config.get('timestamp', 'N/A')}

---

**Note**: This model is intended for research and educational purposes. It should not be used for clinical diagnosis without proper validation and regulatory approval.
"""
    
    return card_content


def push_model_to_hub(output_dir, hub_model_id, private=False):
    """
    Push a trained model to Hugging Face Hub with proper metadata.
    
    Args:
        output_dir: Path to the output directory containing the trained model
        hub_model_id: Full model ID on HF Hub (username/model-name)
        private: Whether to make the repo private
    """
    
    print("="*80)
    print(f"Pushing model to Hugging Face Hub")
    print("="*80)
    print(f"Output Directory: {output_dir}")
    print(f"Hub Model ID: {hub_model_id}")
    print(f"Private: {private}")
    print("="*80)
    
    # Load config
    print("\n[1/6] Loading configuration...")
    config = load_training_config(output_dir)
    is_lora = config.get('use_lora', False)
    
    # Determine model directory
    if is_lora:
        model_dir = os.path.join(output_dir, 'lora_adapter')
    else:
        model_dir = os.path.join(output_dir, 'model')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"✓ Configuration loaded")
    print(f"  Model type: {'LoRA Adapter' if is_lora else 'Full Fine-tuned Model'}")
    print(f"  Base model: {config['model']}")
    print(f"  Classes: {', '.join(config['classes'])}")
    
    # Load test results
    print("\n[2/6] Loading test results...")
    test_results = load_test_results(output_dir)
    if test_results:
        print(f"✓ Test results loaded")
        print(f"  Test Accuracy: {test_results.get('avg_accuracy', 0):.4f}")
        print(f"  Test F1 Score: {test_results.get('avg_f1', 0):.4f}")
    else:
        print("⚠ No test results found")
    
    # Create model card
    print("\n[3/6] Creating model card...")
    model_card = create_model_card(config, test_results)
    
    # Save model card to the model directory
    readme_path = os.path.join(model_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print(f"✓ Model card created and saved to {readme_path}")
    
    # Create repository
    print("\n[4/6] Creating/checking repository on Hugging Face Hub...")
    api = HfApi()
    try:
        repo_url = create_repo(
            repo_id=hub_model_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        return
    
    # Upload model files
    print("\n[5/6] Uploading model files...")
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=hub_model_id,
            repo_type="model",
            commit_message=f"Upload {'LoRA adapter' if is_lora else 'fine-tuned model'}"
        )
        print(f"✓ Model files uploaded successfully")
    except Exception as e:
        print(f"✗ Failed to upload model files: {e}")
        return
    
    # Upload additional files (plots, results)
    print("\n[6/6] Uploading training artifacts...")
    try:
        files_to_upload = [
            'config.json',
            'training_history.csv',
            'test_results.csv',
            'training_curves.png',
            'per_class_metrics.png',
            'confusion_matrix_train.png',
            'confusion_matrix_val.png',
            'confusion_matrix_test.png'
        ]
        
        for filename in files_to_upload:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=f"training_artifacts/{filename}",
                    repo_id=hub_model_id,
                    repo_type="model"
                )
                print(f"  ✓ Uploaded {filename}")
        
        print(f"✓ Training artifacts uploaded")
    except Exception as e:
        print(f"⚠ Warning: Failed to upload some training artifacts: {e}")
    
    print("\n" + "="*80)
    print("✓ SUCCESS! Model pushed to Hugging Face Hub")
    print("="*80)
    print(f"View your model at: https://huggingface.co/{hub_model_id}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Push trained models to Hugging Face Hub with proper metadata"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the training output directory'
    )
    parser.add_argument(
        '--hub_model_id',
        type=str,
        required=True,
        help='Model ID on Hugging Face Hub (format: username/model-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    
    args = parser.parse_args()
    
    # Validate output directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        return
    
    # Validate hub_model_id format
    if '/' not in args.hub_model_id:
        print(f"Error: hub_model_id must be in format 'username/model-name'")
        return
    
    # Push model
    try:
        push_model_to_hub(args.output_dir, args.hub_model_id, args.private)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
