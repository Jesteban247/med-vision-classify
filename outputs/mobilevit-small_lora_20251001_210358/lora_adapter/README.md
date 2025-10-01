---
license: apache-2.0
base_model: apple/mobilevit-small
tags:
- image-classification
- vision
- medical
- brain_tumor
- lora
- transformers
- pytorch
library_name: transformers
datasets:
- brain_tumor
metrics:
- accuracy
- f1
- precision
- recall
- auc
---

# mobilevit-small - Brain_Tumor (LoRA Adapter)

## Model Description

This is a **lora adapter** of [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) fine-tuned on the **Brain_Tumor** dataset for medical image classification.

- **Base Model**: [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)
- **Dataset**: brain_tumor
- **Task**: Multi-class Image Classification
- **Number of Classes**: 4
- **Classes**: glioma, meningioma, no_tumor, pituitary
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

This model is designed for medical image classification, specifically for classifying **brain_tumor** images into the following categories:

- **glioma**
- **meningioma**
- **no_tumor**
- **pituitary**

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
| Loss | 0.1016 |
| Accuracy | 0.9820 |
| Precision | 0.9666 |
| Recall | 0.9661 |
| F1 Score | 0.9657 |
| AUC | 0.9970 |

### Per-Class Performance

| Class | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| glioma | 0.9766 | 1.0000 | 0.9153 | 0.9558 | 0.9967 |
| meningioma | 0.9656 | 0.9128 | 0.9573 | 0.9345 | 0.9931 |
| no_tumor | 0.9969 | 0.9918 | 0.9918 | 0.9918 | 0.9985 |
| pituitary | 0.9891 | 0.9620 | 1.0000 | 0.9806 | 0.9998 |


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

**Training Date**: 2025-10-01 21:04:00

---

**Note**: This model is intended for research and educational purposes. It should not be used for clinical diagnosis without proper validation and regulatory approval.
