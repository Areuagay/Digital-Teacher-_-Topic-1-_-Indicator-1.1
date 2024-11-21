import torch
from models.mosei_mosi_module import MultiModalCNN

# 创建一个MultmodalCNN模型的实例
model = MultiModalCNN(num_classes=8)

# 计算模型的总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")