#!/usr/bin/env python3
"""
简单的PyTorch测试
"""

import sys
print("开始导入测试...")
sys.stdout.flush()

try:
    print("尝试导入torch...")
    sys.stdout.flush()
    import torch
    print("✅ torch导入成功")
    sys.stdout.flush()
    
    print("尝试创建tensor...")
    sys.stdout.flush()
    x = torch.tensor([1, 2, 3])
    print(f"✅ tensor创建成功: {x}")
    sys.stdout.flush()
    
    print("检查CUDA...")
    sys.stdout.flush()
    print(f"CUDA可用: {torch.cuda.is_available()}")
    sys.stdout.flush()
    
except Exception as e:
    print(f"❌ PyTorch错误: {e}")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()

print("测试完成")
sys.stdout.flush()
 
 
 
 