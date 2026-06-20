"""
Inference Script for BGDA
BGDA推理脚本 - 对单张图像或图像列表进行预测

Usage:
    python inference.py --checkpoint CHECKPOINT_PATH --image IMAGE_PATH [--output OUTPUT_PATH]
    python inference.py --checkpoint CHECKPOINT_PATH --image_dir IMAGE_DIR --output_dir OUTPUT_DIR
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bnda.models import DeepLabV3Plus_MS
from bnda.data import ResizeKeepRatio, Normalize_divide, ToTensor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BGDA Inference')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to a single image for inference'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images for batch inference'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for single image prediction'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='predictions',
        help='Output directory for batch predictions'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device id'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Image size [height, width]'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary segmentation'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path, device, num_classes=2, encoder_name='resnet101'):
    """加载模型"""
    model = DeepLabV3Plus_MS(
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=num_classes,
        activation=None
    )
    
    model.to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def preprocess_image(image_path, img_size):
    """预处理单张图像"""
    # 读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # 转换为灰度图并复制为3通道
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([gray, gray, gray], axis=-1)
    
    h, w = img_size
    
    # 应用变换
    transform = transforms.Compose([
        ResizeKeepRatio(h, w),
        Normalize_divide(255),
        ToTensor()
    ])
    
    sample = {"image": image, "mask": np.zeros_like(gray)}
    sample = transform(sample)
    
    return sample['image'].unsqueeze(0), image.shape[:2]  # 添加batch维度，返回原始尺寸


def predict_single_image(model, image_tensor, device, threshold=0.5):
    """预测单张图像"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.max(outputs, 1)[1]
    
    return predictions.cpu().numpy()[0], probabilities.cpu().numpy()[0]


def save_prediction(prediction, original_shape, output_path, color_map=None):
    """保存预测结果"""
    # Resize回原始尺寸
    pred_resized = cv2.resize(
        prediction.astype(np.uint8),
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    if color_map is None:
        # 默认颜色映射：背景=黑色，前景=白色
        color_map = {0: [0, 0, 0], 1: [255, 255, 255]}
    
    # 创建彩色mask
    h, w = pred_resized.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in color_map.items():
        colored_mask[pred_resized == class_idx] = color
    
    # 保存
    cv2.imwrite(output_path, colored_mask)


def main():
    # ==================== 解析参数 ====================
    args = parse_args()
    
    # 验证参数
    if args.image is None and args.image_dir is None:
        raise ValueError("Either --image or --image_dir must be provided")
    
    if args.image is not None and args.output is None:
        args.output = 'prediction.png'
    
    # ==================== 设置设备 ====================
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 加载模型 ====================
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # ==================== 单张图像推理 ====================
    if args.image is not None:
        print(f"\nProcessing single image: {args.image}")
        
        # 预处理
        image_tensor, original_shape = preprocess_image(args.image, args.image_size)
        
        # 预测
        prediction, probabilities = predict_single_image(
            model, image_tensor, device, args.threshold
        )
        
        # 保存结果
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        save_prediction(prediction, original_shape, args.output)
        
        print(f"Prediction saved to: {args.output}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Foreground pixels: {np.sum(prediction > 0)} / {prediction.size}")
    
    # ==================== 批量推理 ====================
    elif args.image_dir is not None:
        if not os.path.isdir(args.image_dir):
            raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = [
            f for f in os.listdir(args.image_dir)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in: {args.image_dir}")
            return
        
        print(f"\nFound {len(image_files)} images in {args.image_dir}")
        print(f"Output directory: {args.output_dir}")
        
        # 批量处理
        for image_file in tqdm(image_files, desc="Processing"):
            image_path = os.path.join(args.image_dir, image_file)
            output_path = os.path.join(args.output_dir, image_file)
            
            try:
                # 预处理
                image_tensor, original_shape = preprocess_image(image_path, args.image_size)
                
                # 预测
                prediction, _ = predict_single_image(
                    model, image_tensor, device, args.threshold
                )
                
                # 保存结果
                save_prediction(prediction, original_shape, output_path)
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        print(f"\nBatch inference completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
