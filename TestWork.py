
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import torch, torchinfo
from torchinfo import summary
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import warnings

import sys, os
sys.path.append(os.path.abspath("../covariate_sim"))
import covariate_generator

class ANN(nn.Module):
    def __init__(self, input_dim = 40, output_dim = 8):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(input_dim, 40)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 32)
        #self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(40, 8)
        #self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.output_layer(x)
        
        return nn.Sigmoid()(x)

def gen_training_detaset():
    models = ["GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB", "IFRGSB",]
    n1 = random.randint(0, 1)
    n2 = random.randint(0, 1)
    test_data = covariate_generator.simulate_dataset(models[n1], 10, 3, True)
    train_data = covariate_generator.simulate_dataset(models[n2], 10, 3, True)
    target_column = ['Target']
    predictor_columns = ['T','FC','x1','x2','x3']
    input_data = np.zeros((40, 1))
    #print(train_data)
    for i in range(10): 
        #print(train_data[:-1,i])
        input_data[i * 4: (i + 1) * 4,0] = train_data[:-1,i]
    training_input = torch.from_numpy(input_data.transpose())
    hmm = [[n1]] * 8;
    #print(hmm)
    training_output = torch.from_numpy(np.array(hmm).transpose())
    #print(target_column)
    #print(predictor_columns)
    #print(X)
    #print(y)
    #print(training_input.shape)
    #print(training_output.shape)

    testing_input = torch.from_numpy(test_data[:4,:].transpose())
    testing_output = torch.from_numpy(test_data[:,4].flatten())
    
    train = torch.utils.data.TensorDataset(training_input, training_output)
    #test = torch.utils.data.TensorDataset(testing_input, testing_output)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
    return train_loader
    
model = ANN()
summary(model)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
epochs = 500
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
# lines 7 onwards
model.train()  # prepare model for training1

for epoch in range(epochs):
    trainloss = 0.0
    valloss = 0.0
    correct = 0
    total = 0
    train_loader = gen_training_detaset()
    for data, target in train_loader:
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(data)
        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct += (predicted == target).sum()
        #print(output)
        #print(output[0, 0].item())
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()*data.size(0)

    trainloss = trainloss/len(train_loader.dataset)
    accuracy = 100 * correct / float(total)
    train_acc_list.append(accuracy)
    train_loss_list.append(trainloss)
    print('Epoch: {} \tTraining Loss: {:.4f}\t Acc: {:.2f}%'.format(
            epoch+1,
            trainloss,
            accuracy
    ))
    epoch_list.append(epoch + 1)
correct = 0
total = 0
valloss = 0
model.eval()

with torch.no_grad():
 for data, target in test_loader:
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)

        output = model(data)
        loss = loss_fn(output, target)
        valloss += loss.item()*data.size(0)

        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct += (predicted == target).sum()

 valloss = valloss/len(test_loader.dataset)
 accuracy = 100 * correct / float(total)
 print(accuracy)
