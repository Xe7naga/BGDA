"""
Custom Loss Functions for Boundary-Guided Segmentation
边界引导分割的自定义损失函数

注意: 实现论文中描述的边界加权BCE-Dice损失和假阳性抑制损失
参数: r=3, α=2.0, τ=0.1, λbw=0.3, λfp=1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.attention_modules import mask_to_boundary, dilate_boundary


def weighted_ce_loss(logits, target, weight_map):
    """
    加权交叉熵损失 (Boundary-weighted BCE)
    
    Args:
        logits: 预测logits [B, C, H, W]
        target: 目标mask [B, H, W]
        weight_map: 权重图 [B, 1, H, W]
    
    Returns:
        加权CE损失
    """
    ce = F.cross_entropy(
        logits,
        target,
        reduction='none'
    )  # [B,H,W]

    ce = ce * weight_map.squeeze(1)
    return ce.mean()


def weighted_dice_loss(logits, target, weight_map, smooth=1e-5):
    """
    加权Dice损失
    
    Args:
        logits: 预测logits [B, C, H, W]
        target: 目标mask [B, H, W]
        weight_map: 权重图 [B, 1, H, W]
        smooth: 平滑项
    
    Returns:
        加权Dice损失
    """
    probs = torch.softmax(logits, dim=1)
    fg_prob = probs[:, 1]               # 前景概率 [B,H,W]
    target_fg = (target > 0).float()
    weight = weight_map.squeeze(1)

    intersection = (fg_prob * target_fg * weight).sum(dim=(1, 2))
    union = (fg_prob * weight).sum(dim=(1, 2)) + (target_fg * weight).sum(dim=(1, 2))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def false_positive_suppression_loss(logits, target, boundary, margin=0.1):
    """
    假阳性抑制损失 (False-positive suppression loss)
    惩罚非边界背景区域的前景预测
    
    Args:
        logits: 预测logits [B, C, H, W]
        target: 目标mask [B, H, W]
        boundary: 边界图 [B, 1, H, W]
        margin: 容差阈值 (τ)
    
    Returns:
        假阳性抑制损失
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # 前景概率
    bg = (target == 0).float()
    non_boundary = (1 - boundary)

    penalty = torch.relu(probs - margin) * bg * non_boundary
    return penalty.mean()


def boundary_weighted_bce_dice_loss(logit_seg, target, 
                                     boundary_radius=3, 
                                     boundary_alpha=2.0,
                                     lambda_bw=0.3,
                                     lambda_fp=1.0,
                                     fp_margin=0.1):
    """
    边界加权BCE-Dice损失 + 假阳性抑制损失
    
    与论文一致的实现:
    - 使用r=3的边界膨胀半径
    - 边界权重系数α=2.0
    - 假阳性抑制容差τ=0.1
    - λbw=0.3, λfp=1.0
    
    Args:
        logit_seg: 分割logits [B, 2, H, W]
        target: 目标mask [B, H, W]
        boundary_radius: r - 边界膨胀半径
        boundary_alpha: α - 边界权重系数
        lambda_bw: λbw - 边界加权损失权重
        lambda_fp: λfp - 假阳性抑制损失权重
        fp_margin: τ - 假阳性抑制容差阈值
    
    Returns:
        总损失
    """
    # 1. 从GT构造boundary权重
    gt_mask = (target > 0).float().unsqueeze(1)   # [B,1,H,W]
    boundary = mask_to_boundary(gt_mask)          # 细边界
    boundary = dilate_boundary(boundary, radius=boundary_radius)  # 边界带 (r=3)

    # 2. 构建边界权重图: weight = 1 + α * boundary
    weight_map = 1.0 + boundary_alpha * boundary  # [B,1,H,W], α=2.0

    # 3. 计算边界加权BCE和Dice损失
    loss_bce = weighted_ce_loss(logit_seg, target, weight_map)
    loss_dice = weighted_dice_loss(logit_seg, target, weight_map)
    
    # 4. 组合边界加权损失
    loss_boundary_weighted = loss_bce + loss_dice

    # 5. 计算假阳性抑制损失
    loss_fp = false_positive_suppression_loss(
        logit_seg, target, boundary, margin=fp_margin
    )

    # 6. 最终加权组合 (λbw=0.3, λfp=1.0)
    total_loss = lambda_bw * loss_boundary_weighted + lambda_fp * loss_fp

    return total_loss


# 保持向后兼容的别名
combined_loss = boundary_weighted_bce_dice_loss
