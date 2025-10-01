---
license: apache-2.0
base_model: apple/mobilevit-small
tags:
- image-classification
- vision
- medical
- chest-xray
- lora
- transformers
- pytorch
library_name: transformers
datasets:
- chest-xray
metrics:
- accuracy
- f1
- precision
- recall
- auc
---

# mobilevit-small - Chest Xray (LoRA Adapter)

## Model Description

This is a **lora adapter** of [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) fine-tuned on the **Chest Xray** dataset for medical image classification.

- **Base Model**: [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)
- **Dataset**: chest-xray
- **Task**: Multi-class Image Classification
- **Number of Classes**: 4
- **Classes**: COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS
- **Training Method**: LoRA (Parameter-Efficient Fine-Tuning)


## LoRA Configuration

This model uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

- **LoRA Rank (r)**: 8
- **LoRA Alpha**: 16
- **LoRA Dropout**: 0.1
- **Target Modules**: query, value
- **Trainable Parameters**: 59,396 (1.19%)
- **Total Parameters**: 4,999,592


## Intended Use

This model is designed for medical image classification, specifically for classifying **chest xray** images into the following categories:

- **COVID19**
- **NORMAL**
- **PNEUMONIA**
- **TURBERCULOSIS**

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

print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
```


## Evaluation Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Loss | 0.0871 |
| Accuracy | 0.9837 |
| Precision | 0.9630 |
| Recall | 0.9682 |
| F1 Score | 0.9656 |
| AUC | 0.9976 |

### Per-Class Performance

| Class | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| COVID19 | 0.9922 | 0.9643 | 0.9818 | 0.9730 | 0.9993 |
| NORMAL | 0.9727 | 0.9367 | 0.9308 | 0.9338 | 0.9946 |
| PNEUMONIA | 0.9740 | 0.9789 | 0.9743 | 0.9766 | 0.9968 |
| TURBERCULOSIS | 0.9961 | 0.9722 | 0.9859 | 0.9790 | 0.9998 |


## Training Details

### Training Hyperparameters

- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Weight Decay**: N/A
- **Seed**: 42

### Model Parameters

- **Total Parameters**: 4,999,592
- **Trainable Parameters**: 59,396 (1.19%)
- **Frozen Parameters**: 4,940,196

### Training Framework

- PyTorch with Hugging Face Transformers
- Accelerate for distributed training
- PEFT for LoRA implementation- LoRA for parameter-efficient fine-tuning

## Limitations and Biases

- This model is trained on a specific medical imaging dataset and may not generalize well to other datasets or imaging modalities
- Medical AI models should always be used as decision support tools, not as standalone diagnostic systems
- Performance may vary depending on image quality, acquisition parameters, and patient demographics
- Always consult with qualified healthcare professionals for medical diagnosis

## Citation

If you use this model, please cite the original base model:

```bibtex
@article{dehghani2021cvt,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

## Model Card Authors

This model card was created as part of a medical image classification project.

## Training Details

**Training Date**: 2025-10-01 21:04:43

---

**Note**: This model is intended for research and educational purposes. It should not be used for clinical diagnosis without proper validation and regulatory approval.
