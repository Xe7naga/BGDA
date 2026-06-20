"""
Test Script for BGDA
BGDA测试脚本 - 在测试集上评估模型性能

Usage:
    python test.py --checkpoint CHECKPOINT_PATH [--test_txts TEST_LIST]
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bnda.config import Config
from bnda.models import DeepLabV3Plus_MS
from bnda.data import SegmentDataset, ResizeKeepRatio, Normalize_divide, ToTensor
from bnda.utils import (
    get_logger,
    combined_loss,
    seg_volume,
    segmentation_metrics
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test BGDA Model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--test_txts',
        type=str,
        nargs='+',
        default=None,
        help='Test data list files (override config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for testing'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device id'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='test_results',
        help='Directory to save test results'
    )
    
    return parser.parse_args()


def test_model(model, device, data_loader, criterion, metric_idxs):
    """
    测试模型
    
    Returns:
        avg_loss, dices, TPVFs, PPVs, volume_dict
    """
    model.eval()
    losses = []
    
    volume_dict = {}
    for metric_idx in metric_idxs:
        volume_dict[metric_idx] = np.array([0, 0, 0, 0])
    
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(data_loader, desc="Testing")):
            inputs = sample_batched['image'].float().to(device)
            labels = sample_batched['mask'].to(device).long()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            # 获取预测
            predictions = torch.max(outputs, 1)[1].cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # 累积体积统计
            for metric_idx in metric_idxs:
                volume_dict[metric_idx] += seg_volume(predictions, labels_cpu, metric_idx)
    
    avg_loss = np.mean(losses)
    dices, TPVFs, PPVs = segmentation_metrics(volume_dict, metric_idxs)
    
    return avg_loss, dices, TPVFs, PPVs, volume_dict


def main():
    # ==================== 解析参数 ====================
    args = parse_args()
    
    # ==================== 加载配置 ====================
    config = Config()
    
    # 覆盖配置
    if args.test_txts is not None:
        config.test_txts = args.test_txts
    config.test_batch = args.batch_size
    config.gpus = args.gpu
    
    # ==================== 设置日志 ====================
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'test.log')
    logger = get_logger(log_path)
    
    logger.info("=" * 80)
    logger.info("BGDA Model Testing")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test datasets: {config.test_txts}")
    logger.info(f"Batch size: {config.test_batch}")
    logger.info(f"GPU: {args.gpu}")
    logger.info("=" * 80)
    
    # ==================== 数据准备 ====================
    test_transforms = transforms.Compose([
        ResizeKeepRatio(config.h, config.w),
        Normalize_divide(255),
        ToTensor()
    ])
    
    test_dataset = SegmentDataset(
        data_txts=config.test_txts,
        transforms=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers
    )
    
    logger.info(f"Number of test images: {len(test_dataset)}")
    
    # ==================== 模型初始化 ====================
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = DeepLabV3Plus_MS(
        encoder_name=config.encoder,
        encoder_weights=None,  # 测试时不需要预训练权重
        classes=config.num_classes,
        activation=None
    )
    
    model.to(device)
    
    # ==================== 加载模型 ====================
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理不同的checkpoint格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        epoch = checkpoint.get('epoch', -1)
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        state_dict = checkpoint
        logger.info("Loaded checkpoint (no epoch info)")
    
    model.load_state_dict(state_dict)
    logger.info("Model loaded successfully!")
    
    # ==================== 测试 ====================
    criterion = combined_loss
    
    logger.info("\nStarting testing...")
    test_loss, test_dices, test_TPVFs, test_PPVs, volume_dict = test_model(
        model, device, test_loader, criterion, config.metric_indexs
    )
    
    # ==================== 输出结果 ====================
    logger.info("\n" + "=" * 80)
    logger.info("Test Results:")
    logger.info("=" * 80)
    logger.info(f"Average Loss: {test_loss:.4f}")
    logger.info(f"\nDice Scores: {test_dices}")
    logger.info(f"Average Dice: {np.mean(test_dices):.4f}")
    logger.info(f"\nTPVF (Recall): {test_TPVFs}")
    logger.info(f"Average TPVF: {np.mean(test_TPVFs):.4f}")
    logger.info(f"\nPPV (Precision): {test_PPVs}")
    logger.info(f"Average PPV: {np.mean(test_PPVs):.4f}")
    logger.info("=" * 80)
    
    # 保存结果到文件
    result_path = os.path.join(args.save_dir, 'test_results.txt')
    with open(result_path, 'w') as f:
        f.write("BGDA Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test datasets: {config.test_txts}\n")
        f.write(f"Average Loss: {test_loss:.4f}\n")
        f.write(f"Dice Scores: {test_dices}\n")
        f.write(f"Average Dice: {np.mean(test_dices):.4f}\n")
        f.write(f"TPVF (Recall): {test_TPVFs}\n")
        f.write(f"Average TPVF: {np.mean(test_TPVFs):.4f}\n")
        f.write(f"PPV (Precision): {test_PPVs}\n")
        f.write(f"Average PPV: {np.mean(test_PPVs):.4f}\n")
    
    logger.info(f"\nResults saved to: {result_path}")
    logger.info("Testing completed!")


if __name__ == '__main__':
    main()
