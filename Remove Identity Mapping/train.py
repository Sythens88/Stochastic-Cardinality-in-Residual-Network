import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
import numpy as np


class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples


class train:
    ## A train class for cifar-10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self, name, model, BATCH_SIZE = 128, LEARNING_RATE = 0.1, MOMENTUM = 0.9, WEIGHT_DECAY = 1e-4, EPOCHS = 200, WARMUP = False):
        transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4)])
        transform_normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        NUM_TRAIN = 45000
        NUM_VAL = 5000
        trainset = datasets.CIFAR10('./data', train=True, download=False, 
                                transform=transforms.Compose([transform_augment, transform_normalize]))
        testset = datasets.CIFAR10('./data', train=False, download=False, transform=transform_normalize)
        valset = datasets.CIFAR10('./data', train=True, download=False, transform=transform_normalize)  
        
        self.train_loader = DataLoader(trainset, batch_size = BATCH_SIZE, sampler = ChunkSampler(NUM_TRAIN))
        self.val_loader = DataLoader(valset, batch_size = BATCH_SIZE, sampler = ChunkSampler(NUM_VAL, start=NUM_TRAIN))
        self.test_loader = DataLoader(testset, batch_size = BATCH_SIZE)
        
        self.name = name
        
        self.epoch = EPOCHS
        self.model = model.to(self.DEVICE)
        self.criterion = nn.CrossEntropyLoss().to(self.DEVICE)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [self.epoch/2,3*self.epoch/4])
        if WARMUP:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE/10
        self.WARMUP = WARMUP

        
        self.train_acc = []
        self.valid_acc = []
        
        self.best_acc = 80
        
        
    def get_param_count(self):
        param_counts = [np.prod(p.size()) for p in self.model.parameters()]
        return sum(param_counts)
    
    def check_accuracy(self, loader):
        num_correct = 0
        num_samples = 0
        #self.model.eval()
        with torch.no_grad():
            for img, label in loader:
                img = img.to(self.DEVICE)
                scores = self.model(img)
                preds = torch.argmax(scores, 1).cpu()
                num_correct += (preds == label).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            self.valid_acc.append(acc)
        
    def train(self,loader):
        num_correct = 0
        num_samples = 0
        #self.model.train()
        for i, (img, label) in enumerate(loader):
            img = img.to(self.DEVICE)
            label = label.to(self.DEVICE)
            scores = self.model(img)
            loss = self.criterion(scores, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            preds = torch.argmax(scores, 1)
            num_correct += (preds == label).sum().item()
            num_samples += preds.size(0)
        
        acc = float(num_correct) / num_samples 
        self.train_acc.append(acc)
        
    def save_model(self):
        if self.valid_acc[-1] >= max(self.valid_acc):
            self.best_acc = self.valid_acc[-1]
            name = "saved_model/" + self.name + ".pkl"
            torch.save(self.model.state_dict(), name )
        
    def main(self):
        ## train
        for epoch in range(self.epoch):
            print('Starting epoch %d / %d' % (epoch+1, self.epoch))
            
            if self.WARMUP and epoch == 10:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.1
            
            self.train(self.train_loader)
            self.model.convert()
            self.check_accuracy(self.val_loader)
            self.model.convert()
            self.lr_scheduler.step()
            self.save_model()
            
        ## save curve
        train_name = 'curve/' + self.name + '_train_acc.npy'
        valid_name = 'curve/' + self.name + '_valid_acc.npy'
        np.save(train_name, self.train_acc)
        np.save(valid_name, self.valid_acc)
        
        ## test
        self.model.load_state_dict(torch.load("saved_model/" + self.name + ".pkl"))
        self.model = self.model.to(self.DEVICE)
        self.model.convert()
        print(self.name, 'Final test accuracy:')
        self.check_accuracy(self.test_loader)

