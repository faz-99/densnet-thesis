import torch
import timm

from utils.load_dataset2 import get_data_loader

train_loader, valid_loader,weights= get_data_loader()
# 创建一个字典来存储每一层的输出
output_dict = {}

# 定义 hook 函数，用于捕获每一层的输出结果
def hook_fn(module, input, output):
    output_dict[module] = output

# 加载 timm 模型
model = timm.create_model('densenet201', pretrained=True)

# 注册钩子，捕获每一层的输出结果
for name, module in model.named_modules():
    module.register_forward_hook(hook_fn)

# 输入张量
input_tensor = torch.randn(1, 3, 224, 224)

# 前向传播，同时捕获每一层的输出结果
for inputs,loader in train_loader:
    i=1
    print(i)
    model(inputs)

    output = model(input_tensor)

    # 打印每一层的输出结果
    for module, output in output_dict.items():
        #    print(module)
        print(output.shape)
    i+1