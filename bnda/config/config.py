"""
Configuration for BGDA Training and Testing

Note: configuration matches the paper description
- CNN/ResNet-based methods use SGD optimizer
- initial learning rate: 1e-3, momentum: 0.9, weight decay: 1e-4
- poly learning rate scheduling with power=0.9
- boundary-weighted BCE-Dice loss + false-positive suppression loss
"""

import os
from pprint import pformat


class Config(object):
    """
    Configuration class to centralize all hyperparameters and path settings.

    Usage:
        config = Config()
        print(pformat(config.__dict__))
    """

    def __init__(self):
        # ==================== Data configuration ====================
        self.train_txts = ["./datalist/exp_train.txt"]  # training list
        self.test_txts = ["./datalist/exp_test.txt"]    # test list
        self.h = 512                                     # image height
        self.w = 512                                     # image width
        self.in_channels = 3                             # input channels
        self.num_classes = 2                             # number of classes (background + foreground)
        self.num_workers = 8                             # DataLoader worker count

        # ==================== Training configuration ====================
        self.train_batch = 16                            # training batch size
        self.test_batch = 16                             # validation batch size
        self.nepoch = 100                                # number of training epochs
        self.lr = 1e-3                                   # initial learning rate (1 × 10^-3)
        self.momentum = 0.9                              # SGD momentum
        self.wd = 1e-4                                   # weight decay (1 × 10^-4)
        self.manualSeed = 666                            # random seed

        # Poly learning rate schedule parameters
        self.poly_power = 0.9                            # poly schedule power
        self.min_lr_ratio = 1e-4                         # minimum learning rate ratio

        # ==================== Model configuration ====================
        self.network = "DeepLabV3+_ms"                   # network name
        self.encoder = "resnet101"                       # encoder type
        self.class_weight = [1, 2]                       # class weights
        self.metric_indexs = [1]                         # class indices for evaluation metrics

        # ==================== Loss configuration ====================
        self.criterion = "boundary_weighted_bce_dice"    # loss type

        # boundary-weighted loss parameters
        self.boundary_radius = 3                         # r: boundary dilation radius
        self.boundary_alpha = 2.0                        # α: boundary weight coefficient
        self.fp_margin = 0.1                             # τ: false-positive suppression tolerance threshold
        self.lambda_bw = 0.3                             # λbw: boundary-weighted loss weight
        self.lambda_fp = 1.0                             # λfp: false-positive suppression loss weight

        # ==================== GPU configuration ====================
        self.gpus = "0"                                  # GPU ids to use

        # ==================== Path configuration ====================
        self.suffix = "resnet101_512x512"                # experiment suffix
        self.log_dir = "logs"                            # log directory
        self.checkpoint_dir = "checkpoint"               # checkpoint directory
        self.summary_dir = "summaries"                   # TensorBoard summary directory

    def get_log_path(self):
        """Get the log file path."""
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
