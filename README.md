# BGDA: Boundary-Guided Dual Attention for Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Introduction

BGDA (Boundary-Guided Dual Attention) is a deep learning-based semantic segmentation framework, particularly suitable for scenarios requiring precise boundary localization, such as industrial defect detection and medical image segmentation.

### Key Features

* **Boundary-Guided Attention Mechanism**: Combines channel attention and spatial attention, leveraging ground-truth boundary information to guide feature enhancement.
* **Multi-Scale Feature Fusion**: Fuses high-level decoder features with low-level encoder features to improve segmentation accuracy.
* **Boundary-Aware Loss Function**: Boundary-weighted BCE-Dice loss combined with false-positive suppression loss.
* **Complete Training-Validation-Testing Pipeline**: Provides an end-to-end solution.

## 🚀 Quick Start

### Requirements

* Python 3.8+
* PyTorch 1.9+
* CUDA 11.0+ (optional but highly recommended)

### Installation

1. **Clone the Repository**

```bash
git clone <repository-url>
cd BGDA
```

2. **Create a Virtual Environment**

```bash
conda create -n bgda python=3.8
conda activate bgda
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Install PyTorch** (choose according to your CUDA version)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📊 Data Preparation

### Data Format

BGDA uses text file lists to manage datasets. Each txt file contains multiple lines, and each line follows the format:

```text
image_path mask_path class_index
```

For example:

```text
/data/images/img001.png /data/masks/img001.png 1
/data/images/img002.png /data/masks/img002.png 1
```

### Create Data Lists

```python
# Example: Create training and testing data lists
import os

def create_data_list(image_dir, mask_dir, output_txt):
    with open(output_txt, 'w') as f:
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.png', '.jpg')):
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)
                if os.path.exists(mask_path):
                    f.write(f"{img_path} {mask_path} 1\n")

# Usage
create_data_list('data/train/images', 'data/train/masks', 'datalist/train.txt')
create_data_list('data/test/images', 'data/test/masks', 'datalist/test.txt')
```

## 🔧 Usage

### 1. Train the Model

```bash
python train.py
```

During training, the framework automatically:

* Saves logs to the `logs/` directory
* Saves model checkpoints to the `checkpoint/` directory
* Records TensorBoard summaries to the `summaries/` directory

View TensorBoard:

```bash
tensorboard --logdir=summaries
```

### 2. Test the Model

```bash
# Use the default test set
python test.py --checkpoint checkpoint/DeepLabV3+_ms/resnet101/resnet101_512x512/epoch100.pth

# Specify a custom test set
python test.py --checkpoint path/to/checkpoint.pth \
               --test_txts datalist/test.txt \
               --batch_size 16 \
               --gpu 0 \
               --save_dir test_results
```

Outputs include:

* Dice Score
* TPVF (True Positive Volume Fraction / Recall)
* PPV (Positive Predictive Value / Precision)

### 3. Inference

#### Single Image

```bash
python inference.py --checkpoint path/to/checkpoint.pth \
                    --image test_image.png \
                    --output prediction.png \
                    --gpu 0
```

#### Batch Inference

```bash
python inference.py --checkpoint path/to/checkpoint.pth \
                    --image_dir test_images/ \
                    --output_dir predictions/ \
                    --gpu 0 \
                    --threshold 0.5
```

## ⚙️ Configuration

All configuration options are defined in `bnda/config/config.py`.

### Training Hyperparameters

CNN/ResNet-based methods use the following defaults:

```python
class Config:
    # Optimizer configuration
    self.lr = 1e-3              # Initial learning rate (1 × 10^-3)
    self.momentum = 0.9         # SGD momentum
    self.wd = 1e-4              # Weight decay (1 × 10^-4)

    # Learning rate schedule
    self.poly_power = 0.9       # poly schedule power

    # Loss function hyperparameters
    self.boundary_radius = 3    # r: boundary dilation radius
    self.boundary_alpha = 2.0   # α: boundary weight coefficient
    self.fp_margin = 0.1        # τ: false-positive suppression tolerance threshold
    self.lambda_bw = 0.3        # λbw: boundary-weighted loss weight
    self.lambda_fp = 1.0        # λfp: false-positive suppression loss weight

    # Other settings
    self.train_batch = 16
    self.nepoch = 100
    self.num_classes = 2
    self.encoder = "resnet101"
```

After updating the configuration, rerun training.

## 🎯 Loss Function

Combined Loss:

```text
Loss = λbw × (Boundary-weighted BCE + Dice) + λfp × False-positive Suppression
```

Where:

* **Boundary-weighted BCE-Dice**: Boundary regions receive higher weights (α=2.0, r=3).
* **False-positive Suppression**: Penalizes false-positive predictions in non-boundary background regions (τ=0.1).
* **λbw=0.3, λfp=1.0**

## 📊 Evaluation Metrics

* **Dice Score**: `2 * |A ∩ B| / (|A| + |B|)`
* **TPVF (Recall)**: `|A ∩ B| / |B|`
* **PPV (Precision)**: `|A ∩ B| / |A|`

Where A denotes the prediction and B denotes the ground truth.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Issues and Pull Requests are welcome!
