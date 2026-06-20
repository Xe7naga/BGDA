"""
Training Script for BGDA
BGDA训练脚本

Usage:
    python train.py [--config CONFIG_PATH]
"""

import os
import sys
import shutil
import random
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
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
    segmentation_metrics,
    log_best_metric
)


def init_worker(worker_id, seed):
    """DataLoader worker初始化函数，确保随机性一致性"""
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, device, data_loader, criterion, optimizer, epoch, writer, metric_idxs):
    """
    训练一个epoch
    
    Returns:
        avg_loss, dices, TPVFs, PPVs
    """
    model.train()
    losses = []
    
    volume_dict = {}
    for metric_idx in metric_idxs:
        volume_dict[metric_idx] = np.array([0, 0, 0, 0])
    
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, sample_batched in enumerate(data_loader):
            inputs = sample_batched['image'].float().to(device)
            labels = sample_batched['mask'].to(device).long()
            
            # 前向传播（训练时传入target用于生成边界图）
            outputs = model(inputs, labels)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()
            optimizer.zero_grad()
            
            # 计算指标
            predictions = torch.max(outputs, 1)[1].cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            for metric_idx in metric_idxs:
                volume_dict[metric_idx] += seg_volume(predictions, labels_cpu, metric_idx)
            
            pbar.update(1)
            pbar.set_description(
                f"Epoch {epoch}, Batch {batch_idx+1}/{len(data_loader)}, "
                f"Train loss: {np.mean(losses):.6f}"
            )
    
    avg_loss = np.mean(losses)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    dices, TPVFs, PPVs = segmentation_metrics(volume_dict, metric_idxs)
    
    return avg_loss, dices, TPVFs, PPVs


def validate_one_epoch(model, device, data_loader, criterion, epoch, writer, metric_idxs):
    """
    验证一个epoch
    
    Returns:
        avg_loss, dices, TPVFs, PPVs
    """
    model.eval()
    losses = []
    
    volume_dict = {}
    for metric_idx in metric_idxs:
        volume_dict[metric_idx] = np.array([0, 0, 0, 0])
    
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(data_loader)):
            inputs = sample_batched['image'].float().to(device)
            labels = sample_batched['mask'].to(device).long()
            
            # 验证时不传入target
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            predictions = torch.max(outputs, 1)[1].cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            for metric_idx in metric_idxs:
                volume_dict[metric_idx] += seg_volume(predictions, labels_cpu, metric_idx)
    
    avg_loss = np.mean(losses)
    writer.add_scalar('test/epoch_loss', avg_loss, epoch)
    
    dices, TPVFs, PPVs = segmentation_metrics(volume_dict, metric_idxs)
    
    return avg_loss, dices, TPVFs, PPVs


def main():
    # ==================== 加载配置 ====================
    config = Config()
    
    # ==================== 设置日志和检查点路径 ====================
    log_path = config.get_log_path()
    if os.path.exists(log_path):
        delete_log = input(f"The log file {log_path} exists, delete it? (y/n): ")
        if delete_log.lower() in ['y', 'yes']:
            os.remove(log_path)
        else:
            log_path = os.path.join(
                config.log_dir, 
                config.network, 
                config.encoder,
                f'{config.suffix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
    
    checkpoint_path = config.get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        delete_ckpt = input(f"The checkpoint folder {checkpoint_path} exists, delete it? (y/n): ")
        if delete_ckpt.lower() in ['y', 'yes']:
            shutil.rmtree(checkpoint_path)
        else:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                config.network,
                config.encoder,
                f'{config.suffix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
    os.makedirs(checkpoint_path, exist_ok=True)
    
    summary_path = config.get_summary_path()
    if os.path.exists(summary_path):
        delete_summary = input(f"The summary folder {summary_path} exists, delete it? (y/n): ")
        if delete_summary.lower() in ['y', 'yes']:
            shutil.rmtree(summary_path)
        else:
            summary_path = os.path.join(
                config.summary_dir,
                config.network,
                config.encoder,
                f'{config.suffix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
    os.makedirs(summary_path, exist_ok=True)
    
    # ==================== 初始化日志 ====================
    logger = get_logger(log_path)
    writer = SummaryWriter(summary_path)
    
    # ==================== 设置随机种子 ====================
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    
    logger.info(f"Random Seed: {config.manualSeed}")
    np.random.seed(config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)
    
    logger.info(str(config))
    
    # ==================== 数据准备 ====================
    train_transforms = transforms.Compose([
        ResizeKeepRatio(config.h, config.w),
        Normalize_divide(255),
        ToTensor()
    ])
    
    test_transforms = transforms.Compose([
        ResizeKeepRatio(config.h, config.w),
        Normalize_divide(255),
        ToTensor()
    ])
    
    train_dataset = SegmentDataset(
        data_txts=config.train_txts,
        transforms=train_transforms
    )
    
    test_dataset = SegmentDataset(
        data_txts=config.test_txts,
        transforms=test_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        worker_init_fn=lambda wid: init_worker(wid, config.manualSeed),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        worker_init_fn=lambda wid: init_worker(wid, config.manualSeed)
    )
    
    logger.info(f"Number of training images: {len(train_dataset)}, test images: {len(test_dataset)}")
    
    # ==================== 模型初始化 ====================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = DeepLabV3Plus_MS(
        encoder_name=config.encoder,
        encoder_weights='imagenet',
        classes=config.num_classes,
        activation=None
    )
    
    torch.backends.cudnn.deterministic = False
    model.to(device)
    
    # ==================== 优化器和调度器 ====================
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.wd
    )
    
    # Poly学习率调度策略 (与论文一致: poly schedule with power 0.9)
    def poly_lr_scheduler(epoch):
        """Poly学习率衰减策略"""
        return (1 - float(epoch) / config.nepoch) ** config.poly_power
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, poly_lr_scheduler)
    
    criterion = combined_loss
    
    # ==================== 训练循环 ====================
    metric_list = []
    
    for epoch in range(config.nepoch):
        logger.info(f"Epoch: {epoch}, Learning rate: {lr_scheduler.get_last_lr()[0]:.10f}")
        
        # 训练
        train_loss, train_dices, train_TPVFs, train_PPVs = train_one_epoch(
            model, device, train_loader, criterion, optimizer, epoch, writer, config.metric_indexs
        )
        
        logger.info(
            f"Epoch: {epoch}, Train Loss: {train_loss:.6f},\n"
            f"Dices: {train_dices}\nTPVF: {train_TPVFs}\nPPV: {train_PPVs}"
        )
        
        # 验证
        test_loss, test_dices, test_TPVFs, test_PPVs = validate_one_epoch(
            model, device, test_loader, criterion, epoch, writer, config.metric_indexs
        )
        
        avg_score = round(np.mean(test_dices), 5)
        metric_list.append(avg_score)
        
        logger.info(
            f"Epoch: {epoch}, Test Loss: {test_loss:.4f},\n"
            f"Dices: {avg_score:.4f}, {test_dices}\n"
            f"TPVF: {test_TPVFs}\nPPV: {test_PPVs}"
        )
        
        # 保存最佳模型
        log_best_metric(
            metric_list, epoch, logger,
            {'epoch': epoch, 'state_dict': model.state_dict()},
            f'{checkpoint_path}/epoch{epoch}.pth',
            save_model=True,
            metric="Dice score"
        )
        
        lr_scheduler.step()
    
    logger.info("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
