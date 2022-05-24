import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, num_hidden3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.tanh=  nn.Linear(num_hidden2, num_hidden3)
        self.linear3 = nn.Linear(num_hidden3, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, x):
        layer1 = self.linear1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.linear2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.tanh(act2)
        act3= self.relu(layer3)
        layer4=self.linear3(act3)
        output = self.sigmoid(layer4)
        return output

def torch_fit(x_tensor, y_true_tensor, model, loss, lr, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    loss1=[]
    for epoch in range(num_epochs):
        optimizer.zero_grad() #to reset the weights and biases before backpropagation 
                                
        y_pred_tensor = model(x_tensor)
        loss_value = loss(y_pred_tensor, y_true_tensor)
        print(f'Epoch {epoch}, loss {loss_value.item():.2f}')
        loss_value.backward() #computes the derivative of the loss w.r.t. the parameters 
        optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.
        loss1.append(loss_value.item())
        plt.plot(loss1)
        plt.show()
    return model



data = pd.read_csv(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\Deep_learning_exercise\01. MLP Binary Classification\data.csv', header=None)

x, y=data.drop(columns=2), data.iloc[:,-1]




x_tensor = torch.tensor(x.values).float()
y_true_tensor = torch.tensor(y.values.reshape(-1,1)).float()
loss = nn.BCELoss()
model = NeuralNetwork(2,10, 10, 2)


train=torch_fit(x_tensor, y_true_tensor, model, loss=loss, lr=0.001, num_epochs=1000)



