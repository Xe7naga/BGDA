"""
Configuration for BGDA Training and Testing
BGDA训练和测试配置

注意: 配置与论文描述一致
- CNN/ResNet-based methods使用SGD优化器
- 初始学习率: 1e-3, momentum: 0.9, weight decay: 1e-4
- poly学习率调度策略, power=0.9
- 边界加权BCE-Dice损失 + 假阳性抑制损失
"""

import os
from pprint import pformat


class Config(object):
    """
    配置类，集中管理所有超参数和路径设置

    使用方法:
        config = Config()
        print(pformat(config.__dict__))
    """

    def __init__(self):
        # ==================== 数据配置 ====================
        self.train_txts = ["./datalist/exp_train.txt"]  # 训练集列表
        self.test_txts = ["./datalist/exp_test.txt"]    # 测试集列表
        self.h = 512                                     # 图像高度
        self.w = 512                                     # 图像宽度
        self.in_channels = 3                             # 输入通道数
        self.num_classes = 2                             # 类别数（背景+前景）
        self.num_workers = 8                             # DataLoader工作进程数

        # ==================== 训练配置 ====================
        self.train_batch = 16                            # 训练batch size
        self.test_batch = 16                             # 验证batch size
        self.nepoch = 100                                # 训练轮数
        self.lr = 1e-3                                   # 初始学习率 (1 × 10^-3)
        self.momentum = 0.9                              # SGD动量
        self.wd = 1e-4                                   # 权重衰减 (1 × 10^-4)
        self.manualSeed = 666                            # 随机种子
        
        # Poly学习率调度参数
        self.poly_power = 0.9                            # poly调度幂次
        self.min_lr_ratio = 1e-4                         # 最小学习率比例

        # ==================== 模型配置 ====================
        self.network = "DeepLabV3+_ms"                   # 网络名称
        self.encoder = "resnet101"                       # 编码器类型
        self.class_weight = [1, 2]                       # 类别权重
        self.metric_indexs = [1]                         # 评估指标对应的类别索引

        # ==================== 损失函数配置 ====================
        self.criterion = "boundary_weighted_bce_dice"    # 损失函数类型
        
        # 边界加权损失参数 (与论文一致)
        self.boundary_radius = 3                         # r: 边界膨胀半径
        self.boundary_alpha = 2.0                        # α: 边界权重系数
        self.fp_margin = 0.1                             # τ: 假阳性抑制容差阈值
        self.lambda_bw = 0.3                             # λbw: 边界加权损失权重
        self.lambda_fp = 1.0                             # λfp: 假阳性抑制损失权重

        # ==================== GPU配置 ====================
        self.gpus = "0"                                  # 使用的GPU编号

        # ==================== 路径配置 ====================
        self.suffix = "resnet101_512x512"                # 实验后缀
        self.log_dir = "logs"                            # 日志目录
        self.checkpoint_dir = "checkpoint"               # 模型检查点目录
        self.summary_dir = "summaries"                   # TensorBoard摘要目录

    def get_log_path(self):
        """获取日志文件路径"""
        log_path = os.path.join(
            self.log_dir,
            self.network,
            self.encoder,
            f'{self.suffix}.log'
        )
        return log_path

    def get_checkpoint_path(self):
        """获取检查点目录路径"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            self.network,
            self.encoder,
            self.suffix
        )
        return checkpoint_path

    def get_summary_path(self):
        """获取TensorBoard摘要路径"""
        summary_path = os.path.join(
            self.summary_dir,
            self.network,
            self.encoder,
            self.suffix
        )
        return summary_path

    def __str__(self):
        """打印配置的字符串表示"""
        return pformat(self.__dict__)
