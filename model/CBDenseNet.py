import torch
import torch.nn as nn
import timm

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

        self.sigmoid_channel = nn.Sigmoid()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x1 = self.fc1(avg_pool) + self.fc2(max_pool)
        x1 = self.sigmoid_channel(x1)
        x2 = x * x1
        x3 = x * self.sigmoid_spatial(x)
        return x2 + x3

class CustomDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super(CustomDenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = self._make_dense_layer(in_channels + i * growth_rate, growth_rate, bn_size)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def _make_dense_layer(self, in_channels, growth_rate, bn_size):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class DenseBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, growth_rate, bn_size=4):
        super(DenseBlockWithCBAM, self).__init__()
        self.dense_block = CustomDenseBlock(in_channels, growth_rate, num_layers, bn_size=bn_size)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.cbam(x)
        return x

class DenseNet201WithCBAM(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNet201WithCBAM, self).__init__()
        self.model = timm.create_model('densenet201', pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        self.dense_block_with_cbam = DenseBlockWithCBAM(in_channels=1920, out_channels=1920, num_layers=32, growth_rate=32)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.dense_block_with_cbam(x)
        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x
if __name__=='__main__':
    model=DenseNet201WithCBAM(num_classes=8)
    print(model)