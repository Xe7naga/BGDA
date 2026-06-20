# BGDA: Boundary-Guided Dual Attention for Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 简介

BGDA (Boundary-Guided Dual Attention) 是一个基于深度学习的语义分割框架，特别适用于需要精确边界定位的场景（如工业缺陷检测、医学图像分割等）。

### 核心特性

- **边界引导注意力机制**: 结合通道注意力和空间注意力，利用GT边界信息引导特征增强
- **多尺度特征融合**: 融合decoder高层特征和encoder低层特征，提升分割精度
- **边界感知损失函数**: 边界加权BCE-Dice损失 + 假阳性抑制损失
- **完整的训练-验证-测试流程**: 提供端到端的解决方案

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，但强烈推荐)

### 安装

1. **克隆仓库**

```bash
git clone <repository-url>
cd BGDA
```

2. **创建虚拟环境**

```bash
conda create -n bgda python=3.8
conda activate bgda
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **安装PyTorch** (根据你的CUDA版本选择)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📊 数据准备

### 数据格式

BGDA使用文本文件列表来管理数据集。每个txt文件包含多行，每行格式为：

```
image_path mask_path class_index
```

例如：
```
/data/images/img001.png /data/masks/img001.png 1
/data/images/img002.png /data/masks/img002.png 1
```

### 创建数据列表

```python
# 示例：创建训练集和测试集列表
import os

def create_data_list(image_dir, mask_dir, output_txt):
    with open(output_txt, 'w') as f:
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.png', '.jpg')):
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)
                if os.path.exists(mask_path):
                    f.write(f"{img_path} {mask_path} 1\n")

# 使用
create_data_list('data/train/images', 'data/train/masks', 'datalist/train.txt')
create_data_list('data/test/images', 'data/test/masks', 'datalist/test.txt')
```

## 🔧 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程中会自动：
- 保存日志到 `logs/` 目录
- 保存模型检查点到 `checkpoint/` 目录
- 记录TensorBoard摘要到 `summaries/` 目录

查看TensorBoard：
```bash
tensorboard --logdir=summaries
```

### 2. 测试模型

```bash
# 使用默认测试集
python test.py --checkpoint checkpoint/DeepLabV3+_ms/resnet101/resnet101_512x512/epoch100.pth

# 指定测试集
python test.py --checkpoint path/to/checkpoint.pth \
               --test_txts datalist/test.txt \
               --batch_size 16 \
               --gpu 0 \
               --save_dir test_results
```

输出包括：
- Dice系数
- TPVF (True Positive Volume Fraction / Recall)
- PPV (Positive Predictive Value / Precision)

### 3. 推理预测

#### 单张图像

```bash
python inference.py --checkpoint path/to/checkpoint.pth \
                    --image test_image.png \
                    --output prediction.png \
                    --gpu 0
```

#### 批量推理

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

After updating the config, rerun training.

## 🎯 Loss Function

Combined loss = λbw × (Boundary-weighted BCE + Dice) + λfp × False-positive Suppression

Where:
- **Boundary-weighted BCE-Dice**: boundary regions receive higher weight (α=2.0, r=3)
- **False-positive Suppression**: penalizes false positive predictions in non-boundary background regions (τ=0.1)
- **λbw=0.3, λfp=1.0**

## 📊 Evaluation Metrics

- **Dice Score**: `2 * |A ∩ B| / (|A| + |B|)`
- **TPVF (Recall)**: `|A ∩ B| / |B|`
- **PPV (Precision)**: `|A ∩ B| / |A|`

Where A is the prediction and B is the ground truth.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## 🤝 Contribution

Issues and Pull Requests are welcome!
