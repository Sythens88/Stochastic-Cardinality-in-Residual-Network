from StochasticCardinality import StochasticCardinality
from ResNeXt import ResNeXt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time

transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4)])
transform_normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10('./data', train=True, download=False, 
                   transform=transforms.Compose([transform_augment, transform_normalize]))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'   
train_loader = DataLoader(trainset, batch_size = 128)
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)





def train(loader):
    for img, label in loader:
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        scores = model(img)
        loss = criterion(scores, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def train_time(model,name): 
    model = model.to(DEVICE)  
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4)

    start_time = time.time()
    for e in range(10):
        for img, label in train_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            scores = model(img)
            loss = criterion(scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()

    print(name,"Train time is", round(end_time-start_time,4))
    


model = ResNeXt(n = 3, d = 16, C = 4, num_class = 10) 
name = "ResNeXt4*16d"
train_time(model,name)

model = StochasticCardinality(d = 16, C = 4, n = 3, dropout = [0,0], num_class = 10).to(DEVICE)
name = "SC4*16d_0"
train_time(model,name)

model = StochasticCardinality(d = 16, C = 4, n = 3, dropout = [0.5,0.5], num_class = 10).to(DEVICE)
name = "SC4*16d_0.5"
train_time(model,name)

print("***********")

model = ResNeXt(n = 3, d = 16, C = 32, num_class = 10) 
name = "ResNeXt32*16d"
train_time(model,name)

model = StochasticCardinality(d = 16, C = 32, n = 3, dropout = [0,0], num_class = 10).to(DEVICE)
name = "SC32*16d_0"
train_time(model,name)

model = StochasticCardinality(d = 16, C = 32, n = 3, dropout = [0.5,0.5], num_class = 10).to(DEVICE)
name = "SC32*16d_0.5"
train_time(model,name)


