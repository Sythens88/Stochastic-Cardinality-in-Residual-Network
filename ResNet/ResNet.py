import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.id_map = nn.Sequential()
        if stride == 2: 
            self.id_map = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2, bias = False),
                nn.BatchNorm2d(out_channels)
            )
            init.kaiming_normal_(self.id_map[0].weight)
        
        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)
    
    def forward(self,x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        identity = self.id_map(x)
        return F.relu(out + identity)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential()
        for i in range(n):
            name = "block" + str(i)
            if i == 0:
                self.layers.add_module(name, ResBlock(in_channels, out_channels))
            else:
                self.layers.add_module(name, ResBlock(out_channels, out_channels))
    
    def forward(self, x):
        return self.layers(x)

class ResNet(nn.Module):
    def __init__(self, n, num_class):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = BasicBlock(16,16,n)
        self.layer2 = BasicBlock(16,32,n)
        self.layer3 = BasicBlock(32,64,n)
        self.classifier = nn.Linear(64,10)
        
        init.kaiming_normal_(self.conv.weight)
        init.kaiming_normal_(self.classifier.weight)
      
    def convert(self):
        pass
        
    def forward(self,x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

