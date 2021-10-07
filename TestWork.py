
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import warnings


data = pd.read_csv("GM31000.csv")
#imputer = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
#imputer.fit(data)
#Transformed_data = imputer.transform(data)
#training_data, testing_data = train_test_split(
#Transformed_data, test_size=0.3, random_state=25)
#print(f"No. of training examples: {training_data.shape}")
#print(f"No. of testing examples: {testing_data.shape}")
target_column = ['Target']
predictor_columns = ['T','FC','x1','x2','x3']
print(target_column)
print(predictor_columns)
X = data[predictor_columns].values
y = data[target_column].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30)
print(X_train.shape)
print(X_test.shape)
class ANN(nn.Module):
    def __init__(self, input_dim = 5, output_dim = 1):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output_layer(x)
        
        return nn.Sigmoid()(x)
model = ANN(input_dim = 5, output_dim = 1)
print(model)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).view(-1, 1)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).view(-1, 1)
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)


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
    for data, target in train_loader:
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(data)
        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct += (predicted == target).sum()

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
