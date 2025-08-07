#!/usr/bin/env python3
print("🔧 简单调试开始")
print("Python运行正常")

import sys
print(f"Python版本: {sys.version}")

import os
print(f"当前目录: {os.getcwd()}")

# 检查文件存在
model_path = "logs/full_collaboration_training/models/500/"
print(f"检查目录: {model_path}")
print(f"目录存在: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"目录内容: {files}")

print("🎉 简单调试完成")
 
 
 
 