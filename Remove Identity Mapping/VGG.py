import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, C, bottleneck = False):
        super(VGGBlock, self).__init__()
        base_width = 16
        hidden_channels = d * int(out_channels/base_width) * C
        stride = 1 if in_channels == out_channels else 2
        self.F = nn.Sequential()
        if bottleneck:
            self.F.add_module('conv1', nn.Conv2d(in_channels, hidden_channels, kernel_size = 1, stride = stride, bias = False))
            self.F.add_module('bn1', nn.BatchNorm2d(hidden_channels))
            self.F.add_module('relu1', nn.ReLU())
            self.F.add_module('conv2', nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, padding = 1, groups = C, bias = False))
            self.F.add_module('bn2', nn.BatchNorm2d(hidden_channels))
            self.F.add_module('relu2', nn.ReLU())
            self.F.add_module('conv3', nn.Conv2d(hidden_channels, out_channels, kernel_size = 1, bias = False))
            self.F.add_module('bn3', nn.BatchNorm2d(out_channels))
            init.kaiming_normal_(self.F[0].weight)
            init.kaiming_normal_(self.F[3].weight)
            init.kaiming_normal_(self.F[6].weight)
        else:
            self.F.add_module('conv1', nn.Conv2d(in_channels, hidden_channels, kernel_size = 3, padding = 1, stride = stride, bias = False))
            self.F.add_module('bn1',nn.BatchNorm2d(hidden_channels))
            self.F.add_module('relu1', nn.ReLU())
            self.F.add_module('conv2', nn.Conv2d(hidden_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, bias = False))
            self.F.add_module('bn2', nn.BatchNorm2d(out_channels))
            init.kaiming_normal_(self.F[0].weight)
            init.kaiming_normal_(self.F[3].weight)
                              
    def forward(self, x):
        out = self.F(x)
        return F.relu(out)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, C, n, bottleneck = False):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential()
        for i in range(n):
            name = "block" + str(i)
            if i == 0:
                self.layers.add_module(name, VGGBlock(in_channels, out_channels, d, C, bottleneck = bottleneck))
            else:
                self.layers.add_module(name, VGGBlock(out_channels, out_channels, d, C, bottleneck = bottleneck))
    
    def forward(self, x):
        return self.layers(x)

class VGG(nn.Module):
    def __init__(self, n, d, C, num_class, bottleneck = False):
        super(VGG, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = BasicBlock(16,16,d,C,n,bottleneck = bottleneck)
        self.layer2 = BasicBlock(16,32,d,C,n,bottleneck = bottleneck)
        self.layer3 = BasicBlock(32,64,d,C,n,bottleneck = bottleneck)
        self.classifier = nn.Linear(64,num_class)
        
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

