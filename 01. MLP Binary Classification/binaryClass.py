from cProfile import label
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, num_hidden3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3=  nn.Linear(num_hidden2, num_hidden3)
        self.linear4 = nn.Linear(num_hidden3, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
    
    def forward(self, x):
        layer1 = self.linear1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.linear2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.linear3(act2)
        act3= self.sigmoid(layer3)
        layer4=self.linear4(act3)
        output = self.sigmoid(layer4)
        return output

def torch_fit(x_tensor, y_tensor, model, loss, lr, num_epochs):

    x_train,x_test,y_train,y_test=train_test_split(x_tensor, y_tensor, random_state=0, test_size=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    train_loss1=[]
    test_losses=[]
    score=[]
    torch.manual_seed(0)
    for epoch in range(num_epochs):
        optimizer.zero_grad() #to reset the weights and biases before backpropagation 
                                
        y_pred_tensor = model(x_train)
        loss_value = loss(y_pred_tensor, y_train)
        #print(f'Epoch {epoch}, loss {loss_value.item():.2f}')
        loss_value.backward() #computes the derivative of the loss w.r.t. the parameters 
        optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.
        train_loss1.append(loss_value.item())
        
        model.eval()
        with torch.no_grad():
            y_preds = model(x_test)
            test_loss = loss(y_preds, y_test)
            results=y_preds>0.5
            #print(results)
            accuracy=sum(results==y_test)/results.shape[0]
            #print(accuracy)
            score.append(accuracy)

            test_losses.append(test_loss.item())
        
        # predict=[]
        # for prediction in y_preds:
        #     if prediction>0.5:
        #         predict.append(1)
        #     else:
        #         predict.append(0)
        # score= (torch.tensor(predict).view(-1,1)==y_test).sum()/len(y_test)
        # print(score)
        model.train()
    #print(score)
            # if predict== y_test:
            #     true_pred=len(predict)
            # else:
            #     false_pred=len(predict)
            # accuracy=true_pred/(true_pred+false_pred)
            #print(f'The Accuracy score of the model is: {accuracy}')
            
        
    plt.plot(train_loss1, label='Train Loss')

    plt.plot(test_losses, label='Test Loss')
    plt.plot(score, label='Accuracy')
    plt.legend()
    plt.show()
    return model





data=pd.read_csv('01. MLP Binary Classification/data.csv', header=None)
x, y=data.drop(columns=2), data.iloc[:,-1]




x_tensor = torch.tensor(x.values).float()
y_tensor = torch.tensor(y.values.reshape(-1,1)).float()
loss = nn.BCELoss()
model = NeuralNetwork(2,10, 20, 30)
#print(model.parameters())


train=torch_fit(x_tensor, y_tensor, model, loss=loss, lr=5e-4, num_epochs=1000)

print(train.parameters()==model.parameters())

