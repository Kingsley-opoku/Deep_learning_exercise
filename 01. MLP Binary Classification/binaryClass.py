import torch
import torch.nn as nn
import pandas as pd
import numpy as np



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        layer1 = self.fc1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.fc2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.fc3(act2)
        output = self.sigmoid(layer3)
        return output



data = pd.read_csv(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\Deep_learning_exercise\01. MLP Binary Classification\data.csv', header=None)

x, y=data.drop(columns=2), data.iloc[:,-1]




x_tensor = torch.tensor(x.values).float()
y_true_tensor = torch.tensor(y.values.reshape(-1,1)).float()
loss = nn.BCELoss()
model = NeuralNetwork(2,100, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
y_pred_tensor = model(x_tensor)
loss_value = loss(y_pred_tensor, y_true_tensor)
#print(f"Initial loss: {loss_value.item():.2f}")



