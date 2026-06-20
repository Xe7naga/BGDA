"""
DeepLabV3+ with Multi-Scale Fusion and Boundary-Guided Attention
带多尺度融合和边界引导注意力的DeepLabV3+模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .attention_modules import (
    BoundaryAwareChannelAttention,
    BoundaryAwareDualAttention,
    MultiScaleFusion,
    BoundaryHead,
    mask_to_boundary,
    dilate_boundary,
    erode_boundary
)


class DeepLabV3Plus_MS(smp.DeepLabV3Plus):
    """
    改进的DeepLabV3+模型，包含：
    1. 多尺度特征融合
    2. 边界感知通道注意力
    3. 可选的边界预测头
    
    Args:
        encoder_name: 编码器名称 (如 'resnet101', 'resnet50')
        encoder_weights: 编码器预训练权重
        classes: 分割类别数
        activation: 激活函数
        use_dual_attention: 是否使用双重注意力（默认使用通道注意力）
    """
    
    def __init__(self,
                 encoder_name='resnet101',
                 encoder_weights='imagenet',
                 classes=2,
                 activation=None,
                 use_dual_attention=False):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )

        # 新增模块
        self.fusion = MultiScaleFusion(decoder_ch=256, low_ch=64)
        
        # 选择注意力机制
        if use_dual_attention:
            self.att = BoundaryAwareDualAttention(256)
        else:
            self.att = BoundaryAwareChannelAttention(256)
        
        # 可选的边界预测头
        self.boundary_head = BoundaryHead(256)

    def forward(self, x, target=None):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            target: 目标mask [B, H, W] (仅在训练时提供，用于生成边界图)
        
        Returns:
            masks: 分割结果 [B, classes, H, W]
            boundary: 边界图 [B, 1, H/4, W/4] (如果启用boundary_head)
        """
        # 1. Encoder
        features = self.encoder(x)
        # features[1]: C2, [B, 256, 128, 128]

        # 2. Decoder
        decoder_feat = self.decoder(features)
        # decoder_feat: [B, 256, 128, 128]

        # 3. Multi-scale fusion (decoder + C2)
        fused = self.fusion(decoder_feat, features[1])
        # fused: [B, 256, 128, 128]

        # ---------- Boundary-aware attention ----------
        boundary_map = None

        if self.training and target is not None:
            gt_mask = (target > 0).float().unsqueeze(1)   # [B,1,H,W]

            # 提取边界
            boundary = mask_to_boundary(gt_mask)          # 细边界
            boundary_band = dilate_boundary(boundary, radius=1)  # 边界带
            
            # 也可以使用内收边界: boundary_band = erode_boundary(gt_mask, radius=1)

            alpha = 1.0  # 边界权重系数
            boundary_map = 1.0 + alpha * boundary_band

            # Resize到fused尺度
            boundary_map = F.interpolate(
                boundary_map,
                size=fused.shape[-2:],
                mode='nearest'
            )

        # 4. Channel/Spatial attention
        fused = self.att(fused, boundary_map)

        # 5. Segmentation head
        masks = self.segmentation_head(fused)
        # masks: [B, classes, H, W]

        # 6. Boundary head (可选)
        # boundary = self.boundary_head(fused)
        
        return masks
