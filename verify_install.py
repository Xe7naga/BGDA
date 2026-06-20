"""
Quick verification script to test BGDA package imports
快速验证脚本，测试BGDA包的导入是否正常
"""

import sys
import os

def test_imports():
    """测试所有模块的导入"""
    print("=" * 80)
    print("Testing BGDA Package Imports")
    print("=" * 80)
    
    tests = [
        ("Config", "from bnda.config import Config"),
        ("DeepLabV3Plus_MS", "from bnda.models import DeepLabV3Plus_MS"),
        ("SegmentDataset", "from bnda.data import SegmentDataset"),
        ("Transforms", "from bnda.data import ResizeKeepRatio, Normalize_divide, ToTensor"),
        ("Losses", "from bnda.utils import combined_loss"),
        ("Metrics", "from bnda.utils import seg_volume, segmentation_metrics"),
        ("Logger", "from bnda.utils import get_logger"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:30s} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name:30s} - FAILED: {e}")
            failed += 1
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


def test_model_creation():
    """测试模型创建"""
    print("\nTesting Model Creation...")
    print("-" * 80)
    
    try:
        import torch
        from bnda.models import DeepLabV3Plus_MS
        
        # 创建模型
        model = DeepLabV3Plus_MS(
            encoder_name='resnet18',  # 使用小模型快速测试
            encoder_weights=None,
            classes=2,
            activation=None
        )
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model created successfully")
        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("-" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print("-" * 80)
        return False


def main():
    """主函数"""
    print("\n🔍 BGDA Package Verification\n")
    
    # 测试导入
    import_ok = test_imports()
    
    # 测试模型创建（如果导入成功）
    if import_ok:
        model_ok = test_model_creation()
    else:
        model_ok = False
    
    # 总结
    print("\n" + "=" * 80)
    if import_ok and model_ok:
        print("✅ All tests passed! BGDA package is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
