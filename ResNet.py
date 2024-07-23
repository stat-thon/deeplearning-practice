
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    
    def __init__(self):
        super(ResNet, self).__init_()
        
        # filter size 7
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3) # in_channels self.parameter로 추가하는 게 낫지 않을까
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        # 64 channels
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1) # stride default = 1
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()
        
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()
        
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.relu2_3 = nn.ReLU()
        
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.relu2_4 = nn.ReLU()
        
        self.conv2_5 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2_5 = nn.BatchNorm2d(64)
        self.relu2_5 = nn.ReLU()
        
        self.conv2_6 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2_6 = nn.BatchNorm2d(64)
        self.relu2_6 = nn.ReLU()
        
        # 128 channels
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU()
        
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU()
        
        self.conv3_proj = nn.Conv2d(64, 128, kernel_size = 1) # for projection shortcut
        
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.relu3_3 = nn.ReLU()
        
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_4 = nn.BatchNorm2d(128)
        self.relu3_4 = nn.ReLU()
        
        self.conv3_5 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_5 = nn.BatchNorm2d(128)
        self.relu3_5 = nn.ReLU()
        
        self.conv3_6 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_6 = nn.BatchNorm2d(128)
        self.relu3_6 = nn.ReLU()
        
        self.conv3_7 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_7 = nn.BatchNorm2d(128)
        self.relu3_7 = nn.ReLU()
        
        self.conv3_8 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn3_8 = nn.BatchNorm2d(128)
        self.relu3_8 = nn.ReLU()
        
        # 256 channels
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU()
        
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU()
        
        self.conv4_proj = nn.Conv2d(128, 256, kernel_size = 1) # for projection shortcut
        
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.ReLU()
        
        self.conv4_4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_4 = nn.BatchNorm2d(256)
        self.relu4_4 = nn.ReLU()
        
        self.conv4_5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_5 = nn.BatchNorm2d(256)
        self.relu4_5 = nn.ReLU()
        
        self.conv4_6 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_6 = nn.BatchNorm2d(256)
        self.relu4_6 = nn.ReLU()
        
        self.conv4_7 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_7 = nn.BatchNorm2d(256)
        self.relu4_7 = nn.ReLU()
        
        self.conv4_8 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_8 = nn.BatchNorm2d(256)
        self.relu4_8 = nn.ReLU()
        
        self.conv4_9 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_9 = nn.BatchNorm2d(256)
        self.relu4_9 = nn.ReLU()
        
        self.conv4_10 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_10 = nn.BatchNorm2d(256)
        self.relu4_10 = nn.ReLU()
        
        self.conv4_11 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_11 = nn.BatchNorm2d(256)
        self.relu4_11 = nn.ReLU()
        
        self.conv4_12 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn4_12 = nn.BatchNorm2d(256)
        self.relu4_12 = nn.ReLU()
        
        # 512 channels
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU()
        
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU()
        
        self.conv5_proj = nn.Conv2d(256, 512, kernel_size = 1) # for projection shortcut
        
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU()
        
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.relu5_4 = nn.ReLU()
        
        self.conv5_5 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn5_5 = nn.BatchNorm2d(512)
        self.relu5_5 = nn.ReLU()
        
        self.conv5_6 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn5_6 = nn.BatchNorm2d(512)
        self.relu5_6 = nn.ReLU()
        
        # global avg pooling
        self.Adaavgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC layer
        self.fc1 = nn.Linear(512, 1000)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.maxpool1(x)
        
        # 64 channels
        identity = x # input 기록
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x) + identity # 차원이 동일
        
        identity = x # input 기록
        
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x) + identity
        
        identity = x # input 기록
        
        x = self.conv2_5(x)
        x = self.bn2_5(x)
        x = self.relu2_5(x)
        
        x = self.conv2_6(x)
        x = self.bn2_6(x)
        x = self.relu2_6(x)
        
        identity = x # input 기록
        
        # 128 channels
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        y = self.conv3_proj(identity) # 1x1 conv filter
        x = self.relu3_2(x + y)
        
        identity = x # input 기록
        
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu3_3(x)
        
        x = self.conv3_4(x)
        x = self.bn3_4(x)
        x = self.relu3_4(x) + identity
        
        identity = x # input 기록
        
        x = self.conv3_5(x)
        x = self.bn3_5(x)
        x = self.relu3_5(x)
        
        x = self.conv3_6(x)
        x = self.bn3_6(x)
        x = self.relu3_6(x) + identity
        
        identity = x # input 기록
        
        x = self.conv3_7(x)
        x = self.bn3_7(x)
        x = self.relu3_7(x)
        
        x = self.conv3_8(x)
        x = self.bn3_8(x)
        x = self.relu3_8(x) + identity
        
        identity = x # input 기록
        
        # 256 channels
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        y = self.conv4_proj(identity) # 1x1 filter 256개 사용
        x = self.relu4_2(x + y)
        
        identity = x
        
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu4_3(x)
        
        x = self.conv4_4(x)
        x = self.bn4_4(x)
        x = self.relu4_4(x) + identity
        
        identity = x
        
        x = self.conv4_5(x)
        x = self.bn4_5(x)
        x = self.relu4_5(x)
        
        x = self.conv4_6(x)
        x = self.bn4_6(x)
        x = self.relu4_6(x) + identity
        
        identity = x
        
        x = self.conv4_7(x)
        x = self.bn4_7(x)
        x = self.relu4_7(x)
        
        x = self.conv4_8(x)
        x = self.bn4_8(x)
        x = self.relu4_8(x) + identity
        
        identity = x
        
        x = self.conv4_9(x)
        x = self.bn4_9(x)
        x = self.relu4_9(x)
        
        x = self.conv4_10(x)
        x = self.bn4_10(x)
        x = self.relu4_10(x) + identity
        
        identity = x
        
        x = self.conv4_11(x)
        x = self.bn4_11(x)
        x = self.relu4_11(x)
        
        x = self.conv4_12(x)
        x = self.bn4_12(x)
        x = self.relu4_12(x) + identity
        
        identity = x
        
        # 512 channels
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        y = self.conv5_proj(identity)
        x = self.relu5_2(x + y)
        
        identity = x
        
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu5_3(x)
        
        x = self.conv5_4(x)
        x = self.bn5_4(x)
        x = self.relu5_4(x) + identity
        
        identity = x
        
        x = self.conv5_5(x)
        x = self.bn5_5(x)
        x = self.relu5_5(x)
        
        x = self.conv5_6(x)
        x = self.bn5_6(x)
        x = self.relu5_6(x) + identity
        
        # global avg pool 2d
        x = self.Adaavgpool1(x)
        
        # Linear
        x = self.fc1(x)
        
        return x
