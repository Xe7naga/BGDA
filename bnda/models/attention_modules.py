"""
Boundary-Guided Dual Attention Modules
边界引导双注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareChannelAttention(nn.Module):
    """
    边界感知通道注意力模块
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, boundary_map=None):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            boundary_map: 边界图 [B, 1, H, W] (可选)
        
        Returns:
            增强后的特征
        """
        b, c, h, w = x.size()

        # Global average pooling
        pooled = x.mean(dim=(2, 3))
        y = self.fc(pooled).view(b, c, 1, 1)

        if boundary_map is not None:
            # 只在边界区域增强，非边界区域保持原样
            boundary_map = boundary_map.detach()
            return x + x * y * boundary_map
        else:
            return x * y


class BoundaryAwareDualAttention(nn.Module):
    """
    边界感知双重注意力模块 (通道 + 空间注意力)
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel Attention
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention (CBAM style)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Fusion convolution
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, boundary_map=None):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            boundary_map: 边界图 [B, 1, H, W] (可选)
        
        Returns:
            增强后的特征
        """
        b, c, h, w = x.size()

        if boundary_map is not None:
            boundary_map = F.interpolate(
                boundary_map,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            boundary_map = boundary_map.detach()

        # ---------------------------
        # Channel Attention Branch
        # ---------------------------
        pooled = x.mean(dim=(2, 3))
        channel_att = self.channel_fc(pooled).view(b, c, 1, 1)
        Fc = x * channel_att

        if boundary_map is not None:
            Fc = x + Fc * boundary_map
        else:
            Fc = x + Fc

        # ---------------------------
        # Spatial Attention Branch
        # ---------------------------
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_conv(spatial_input)
        Fs = x * spatial_att

        if boundary_map is not None:
            Fs = x + Fs * boundary_map
        else:
            Fs = x + Fs

        # ---------------------------
        # Fusion
        # ---------------------------
        F_att = Fc + Fs
        F_att = self.fusion_conv(F_att)

        # ---------------------------
        # Residual connection
        # ---------------------------
        out = F_att + x

        return out


class MultiScaleFusion(nn.Module):
    """
    多尺度特征融合模块
    
    Args:
        decoder_ch: Decoder特征通道数
        low_ch: 低层特征通道数
    """
    def __init__(self, decoder_ch=256, low_ch=64):
        super().__init__()
        self.low_proj = nn.Conv2d(low_ch, decoder_ch, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_ch * 2, decoder_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, decoder_feat, low_feat):
        """
        Args:
            decoder_feat: Decoder特征 [B, decoder_ch, H, W]
            low_feat: 低层特征 [B, low_ch, H', W']
        
        Returns:
            融合后的特征 [B, decoder_ch, H, W]
        """
        low_feat = self.low_proj(low_feat)
        
        # Resize to match decoder feature size
        low_feat = nn.functional.interpolate(
            low_feat,
            size=decoder_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        x = torch.cat([decoder_feat, low_feat], dim=1)
        return self.fuse(x)


class BoundaryHead(nn.Module):
    """
    边界预测头
    
    Args:
        in_ch: 输入通道数
    """
    def __init__(self, in_ch=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, in_ch, H, W]
        
        Returns:
            边界图 [B, 1, H, W]
        """
        return self.head(x)


def mask_to_boundary(mask, threshold=0.1):
    """
    从mask中提取边界
    
    Args:
        mask: 二值mask [B, 1, H, W] (0/1)
        threshold: 梯度阈值
    
    Returns:
        边界图 [B, 1, H, W] (0/1)
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=torch.float32,
        device=mask.device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=torch.float32,
        device=mask.device
    ).unsqueeze(0)

    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)

    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary = (grad > threshold).float()
    
    return boundary


def dilate_boundary(boundary, radius=1):
    """
    边界膨胀（外扩）
    
    Args:
        boundary: 边界图 [B, 1, H, W]
        radius: 膨胀半径
    
    Returns:
        膨胀后的边界图
    """
    kernel = torch.ones(
        1, 1, 2 * radius + 1, 2 * radius + 1,
        device=boundary.device
    )
    dilated = F.conv2d(boundary, kernel, padding=radius)
    return (dilated > 0).float()


def erode_boundary(mask, radius=1):
    """
    边界腐蚀（内收）
    
    Args:
        mask: 二值mask [B, 1, H, W]
        radius: 腐蚀半径
    
    Returns:
        内收边界带
    """
    kernel_size = 2 * radius + 1
    kernel = torch.ones(
        1, 1, kernel_size, kernel_size,
        device=mask.device
    )

    conv = F.conv2d(mask, kernel, padding=radius)
    eroded = (conv == kernel.numel()).float()

    # Inner boundary band
    inner_boundary = mask - eroded
    return inner_boundary
