"""
Evaluation Metrics for Segmentation
分割评估指标
"""

import os
import numpy as np
import torch


def seg_volume(batch_output, batch_label, metric_idx):
    """
    计算单个类别的分割体积统计
    
    Args:
        batch_output: 预测结果 [B, H, W]
        batch_label: 真实标签 [B, H, W]
        metric_idx: 类别索引
    
    Returns:
        numpy array: [union, intersection, volume_gt, volume_pred]
    """
    mask_o = (batch_output == metric_idx)
    mask_y = (batch_label == metric_idx)
    
    inter = (mask_o * mask_y).sum()
    union = mask_o.sum() + mask_y.sum()
    v_y = mask_y.sum()
    v_o = mask_o.sum()

    return np.array([union, inter, v_y, v_o])


def segmentation_metrics(volume_dict, metric_idxs=[1]):
    """
    计算分割指标：Dice, TPVF, PPV
    
    Args:
        volume_dict: 体积统计字典 {metric_idx: [union, inter, v_y, v_o]}
        metric_idxs: 需要计算的类别索引列表
    
    Returns:
        dices: Dice系数列表
        TPVFs: True Positive Volume Fraction列表
        PPVs: Positive Predictive Value列表
    """
    dices, TPVFs, PPVs = [], [], []
    
    for metric_idx in metric_idxs:
        if metric_idx not in volume_dict:
            continue
            
        [union, inter, v_y, v_o] = list(volume_dict[metric_idx])

        dice = 0 if inter == 0 else round(float(2 * inter) / union, 5)
        TPVF = 0 if v_y == 0 else round(float(inter) / v_y, 5)
        PPV = 0 if v_o == 0 else round(float(inter) / v_o, 5)
        
        dices.append(dice)
        TPVFs.append(TPVF)
        PPVs.append(PPV)

    return dices, TPVFs, PPVs


def log_best_metric(metric_list, cur_epoch_idx, logger, state, save_path, 
                   save_model=True, metric="Dice"):
    """
    记录并保存最佳指标
    
    Args:
        metric_list: 历史指标列表
        cur_epoch_idx: 当前epoch索引
        logger: 日志对象
        state: 模型状态字典
        save_path: 保存路径
        save_model: 是否保存模型
        metric: 指标名称
    """
    if len(metric_list) == 0:
        return
    
    best_idx = np.argmax(metric_list)
    best_metric = metric_list[best_idx]
    
    if best_idx == cur_epoch_idx:
        logger.info(f"Epoch: {cur_epoch_idx}, Validation {metric} improved to {best_metric:.4f}")
        if save_model:
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(state, save_path)
            logger.info(f"Model saved in file: {save_path}")
    else:
        logger.info(
            f"Epoch: {cur_epoch_idx}, Validation {metric} didn't improve. "
            f"Best is {best_metric:.4f} in epoch {best_idx}"
        )
