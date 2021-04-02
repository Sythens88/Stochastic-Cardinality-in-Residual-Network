import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, d, bottleneck = False):
        super(Block, self).__init__()
        base_width = 16
        hidden_channels = d * int(out_channels/base_width)
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
        return out

class SCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, C, dropout, bottleneck = False):
        super(SCBlock, self).__init__()
        self.train = True
        self.C = C
        self.dropout = dropout
        self.Blocks = nn.ModuleList([Block(in_channels, out_channels, d, bottleneck = bottleneck) for _ in range(C)])
        
        self.id_map = nn.Sequential()
        if in_channels != out_channels:
            self.id_map = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2, bias = False),
                nn.BatchNorm2d(out_channels)
            )
            init.kaiming_normal_(self.id_map[0].weight)
        
    def forward(self, x):
        ## train time
        if self.train:
            ## decide whether to drop the branch: 0 means drop and 1 means not drop
            drop = [int(random.random() > self.dropout) for _ in range(self.C)]
            ## add all the branches which are not dropped
            flag = False
            for i in range(self.C):
                if drop[i] == 1:
                    if not flag:
                        out = self.Blocks[i](x)
                        flag = True
                    else:
                        out += self.Blocks[i](x)
            ## identity mapping
            if flag:
                out += self.id_map(x)
            else:
                out = self.id_map(x)
        ## test time
        else:
            for i in range(self.C):
                if i == 0:
                    out = self.Blocks[i](x)*(1-self.dropout)
                else:
                    out += self.Blocks[i](x)*(1-self.dropout)
            out += self.id_map(x)
        return F.relu(out)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, C, n, dropout, bottleneck = False):
        super(BasicBlock, self).__init__()
        ## calculate dropout step
        drop_step = (dropout[1]-dropout[0])/(n-1)
        self.n = n
        self.layers = nn.Sequential()
        
        
        for i in range(n):
            name = 'layer' + str(i)
            if i == 0:
                self.layers.add_module(name, SCBlock(in_channels, out_channels, d, C, dropout[0], bottleneck = bottleneck))
            else:
                self.layers.add_module(name, SCBlock(out_channels, out_channels, d, C, dropout[0]+i*drop_step, bottleneck = bottleneck))
        
    def convert(self):
        for i in range(self.n):
            self.layers[i].train = bool(1-self.layers[i].train)
    
    def forward(self, x):
        return self.layers(x)

class StochasticCardinality(nn.Module):
    def __init__(self, d, C, n, dropout, num_class, bottleneck = False):
        super(StochasticCardinality, self).__init__()
        self.train = True
        self.conv = nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(16)
        
        drop_step = (dropout[1]-dropout[0])/3
        self.layer1 = BasicBlock(16,16,d,C,n,[dropout[0],dropout[0]+drop_step], bottleneck = bottleneck)
        self.layer2 = BasicBlock(16,32,d,C,n,[dropout[0]+drop_step,dropout[0]+2*drop_step], bottleneck = bottleneck)
        self.layer3 = BasicBlock(32,64,d,C,n,[dropout[0]+2*drop_step,dropout[1]], bottleneck = bottleneck)
        
        self.classifier = nn.Linear(64, num_class)
        
        init.kaiming_normal_(self.conv.weight)
        init.kaiming_normal_(self.classifier.weight)
        
    def convert(self):
        self.train = bool(1-self.train)
        self.layer1.convert()
        self.layer2.convert()
        self.layer3.convert()
        
        
    def forward(self,x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        
        return x

