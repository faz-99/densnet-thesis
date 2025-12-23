import timm
import torch
from torch import nn

import config_clean as config
from acb import ACBlock
from model.SENet import SELayer

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
def getModel():
    model = timm.create_model('densenet201', pretrained=True, num_classes=config.class_num)

    dense_layers = []
    for name, module in model.named_modules():
        if isinstance(module, timm.models.densenet.DenseBlock):
            dense_layers.append(module)

    # print(len(dense_layers))
    # num_features = model.features.denseblock1.denselayer1.norm2.num_features
    # print(num_features)
    label = [3, 5, 7, 9]
    for i, dense_layer in enumerate(dense_layers):
        for j in dense_layer:
            num_features = model.features[label[i]][f'{j}'].norm2.num_features
            model.features[label[i]][f'{j}'].add_module("SELayer", SELayer(num_features))

    transition_conv1 = model.features.transition1.norm
    se_layer = SELayer(transition_conv1.num_features)
    transition_conv1.add_module("SELayer", se_layer)
    #
    transition_conv2 = model.features.transition2.norm
    se_layer2 = SELayer(transition_conv2.num_features)
    transition_conv2.add_module("SELayer", se_layer2)
    #
    transition_conv3 = model.features.transition3.norm
    se_layer3 = SELayer(transition_conv3.num_features)
    transition_conv3.add_module("SELayer", se_layer3)

    # in_channels1 = model.features.denseblock4.denselayer32.conv2.out_channels
    # # print(in_channels1)
    # # out_channels1=128
    #
    # norm = model.features.norm5
    # acb = ACBlock(in_channels1, 32, kernel_size=3, padding=1, stride=1, deploy=False)
    # norm.add_module('asymmentric_conv', acb)

    # out_channels=model.features.denseblock4.denselayer32.conv2.out_channels
    # out_channels=model.features.num_features
    return model


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class class_model(nn.Module):
    def __init__(self):
        super(class_model,self).__init__()
        self.densenet=getModel()

        self.dropout1=nn.Dropout(p=0.5)
        self.conv2d_1=nn.Conv2d(in_channels=512, out_channels=1792, kernel_size=2, stride=2)
        self.aff1=iAFF(channels=1792)
        self.dropout2=nn.Dropout(p=0.5)
        self.conv2d_2=nn.Conv2d(in_channels=1792, out_channels=1920, kernel_size=2, stride=2)
        self.aff2=iAFF(channels=1920)

        self.lstm = nn.LSTM(1920, 128, batch_first=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, config.class_num)
    def forward(self,x):
        x=self.densenet.features.conv0(x)
        x=self.densenet.features.norm0(x)
        x=self.densenet.features.pool0(x)
        x=self.densenet.features.denseblock1(x)
        x=self.densenet.features.transition1(x)
        x=self.densenet.features.denseblock2(x)
        #y=self.dropout1(x)
        y=self.conv2d_1(x)
        x=self.densenet.features.transition2(x)
        x=self.densenet.features.denseblock3(x)
        x=self.aff1(x,y)
        #y1=self.dropout2(x)
        y1=self.conv2d_2(x)
        x=self.densenet.features.transition3(x)
        x=self.densenet.features.denseblock4(x)
        x=self.aff2(x,y1)
        x=self.densenet.features.norm5(x)

        # 调整特征维度适应LSTM模型
        features = self.pool(x)
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, width, channels)  # 32,1,1920
        # 将特征传入LSTM模型
        lstm_output, _ = self.lstm(features)
        # 只取最后一个时间步的输出
        x = lstm_output[:, -1, :]
        x=self.fc(x)
        return x
#
# if __name__=="__main__":
#     input_dim = getModel().num_features
#     hidden_dim = 50
#     layer_dim = 2
#     out_dim = 64
#     output_size = config.class_num
#     model = ClassificationModel(input_dim, hidden_dim, layer_dim, out_dim=output_size)
#     #model=ClassificationModel(out_dim,config.class_num)
#     input=torch.randn(32,3,224,224)
#     out= model(input)
#     print(out.shape)
#     print(model)