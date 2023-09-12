import torch
import torch.nn as nn 
from math import floor 
from tqdm import tqdm 

from matplotlib import pyplot as plt 

class CNN(nn.Module):
    def __init__(self, device, input_shape):
        super(CNN, self).__init__()

        self.log = {
            'training loss' : [],
            'validation loss' : [],
            'training accuracy' : [],
            'validation accuracy' : []
        } 
        
        k, m, n = input_shape 
        self.device = device 
         
        self.maxpool = nn.MaxPool2d(2 ,2)
        self.relu = nn.ReLU() 
        self.flatten = nn.Flatten() 
        self.dropout = nn.Dropout() 
        self.softmax = nn.Softmax() 
         
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
         
        self.linear1 = nn.Linear(in_features=128 * floor((m - 8) / 2) * floor((n - 8) / 2), out_features=64) 
        self.linear2 = nn.Linear(in_features=64, out_features=32) 
        self.linear3 = nn.Linear(in_features=32, out_features=3) 

    def forward(self, x):
        y_hat = self.relu(self.conv1(x)) 
        y_hat = self.relu(self.conv2(y_hat)) 
        y_hat = self.relu(self.conv3(y_hat)) 
        y_hat = self.relu(self.conv4(y_hat)) 

        y_hat = self.flatten(self.maxpool(y_hat))
         
        y_hat = self.relu(self.linear1(y_hat)) 
        y_hat = self.dropout(y_hat) 
        y_hat = self.relu(self.linear2(y_hat)) 
        y_hat = self.softmax(self.linear3(y_hat)) 
        return y_hat

    def fit(self, num_epochs, lr, loader_train, loader_val):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss() 
         
        for i in tqdm(range(num_epochs)):
            for xs, ys in loader_train: 
                xs = xs.to(self.device) 
                ys = ys.to(self.device) 
                optimizer.zero_grad()   
                y_hats = self(xs)
                loss = criterion(y_hats, ys) 
                loss.backward()
                optimizer.step()    
                self.log['training loss'].append(loss.item()) 
         
            self.eval() 

            with torch.no_grad():
                for xs, ys in loader_val:  
                    xs = xs.to(self.device)
                    ys = ys.to(self.device) 
                    y_hats = self(xs)
                    loss = criterion(y_hats, ys) 
                    self.log['validation loss'].append(loss.item()) 

                total_instances = 0 
                total_correct_preds = 0 
                for xs, ys in loader_train: 
                    xs = xs.to(self.device)
                    ys = ys.to(self.device) 
                     
                    classifications = torch.argmax(self(xs), dim=1)
                    correct_preds = len([classifications[i] for i in range(0, len(ys)) if ys[i][classifications[i]] == 1]) 
                    total_instances += len(ys) 
                    total_correct_preds += correct_preds 
                self.log['training accuracy'].append(total_correct_preds / total_instances) 
                  
                total_instances = 0 
                total_correct_preds = 0 
                for xs, ys in loader_val: 
                    xs = xs.to(self.device)
                    ys = ys.to(self.device) 
                     
                    classifications = torch.argmax(self(xs), dim=1)
                    correct_preds = len([classifications[i] for i in range(0, len(ys)) if ys[i][classifications[i]] == 1]) 
                    total_instances += len(ys) 
                    total_correct_preds += correct_preds 
                self.log['validation accuracy'].append(total_correct_preds / total_instances) 
                 
            self.train() 

    def evaluate(self):
        plt.figure(figsize=(30, 20)) 
        plt.subplot(2, 1, 1) 
        plt.plot(self.log['training loss'])
        plt.title('training loss') 
        plt.xlabel('batch') 
        plt.ylabel('loss') 
         
        plt.subplot(4, 1, 2) 
        plt.plot(self.log['validation loss'])
        plt.title('validation loss') 
        plt.xlabel('batch') 
        plt.ylabel('loss') 
         
        plt.subplot(4, 1, 3) 
        plt.plot(self.log['training accuracy']) 
        plt.title('training accuracy') 
        plt.xlabel('epoch') 
        plt.ylabel('accuracy') 
         
        plt.subplot(4, 1, 4) 
        plt.plot(self.log['validation accuracy']) 
        plt.title('validation accuracy') 
        plt.xlabel('epoch') 
        plt.ylabel('accuracy') 
        plt.show() 