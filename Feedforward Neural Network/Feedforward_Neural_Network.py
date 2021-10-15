import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn

class NeuralNet(object):
    
    def __init__(self,n,p,N):
        self.n = n   # Number of features (1 for univariate problem)
        self.p = p   # Number of nodes in the hidden layer
        self.N = N   # Number of outputs (1 for the regression problem)
        
        # Instantiate weight matrices
        self.W_1 = torch.randn(n,p)
        self.W_2 = torch.randn(p,N)
        
        # Instantiate bias vectors (Why do we need this?)
        self.b_1 = torch.randn(1,p)
        self.b_2 = torch.randn(1,N)
        
        ### CHANGE FROM ABOVE ###  
        # Collect the model parameters, and tell pytorch to
        # collect gradient information about them.
        self.parameters = [self.W_1,self.W_2,self.b_1,self.b_2]
        for param in self.parameters:
            param.requires_grad_()
    def forward(self,X):
        # Applies the neural network model
        ## All of these self. prefixes save calculation results
        ## as class variables - we can inspect them later if we
        ## wish to
        self.X = X
        self.z = self.X @ self.W_1 + self.b_1  # First linear 
        self.h = torch.sigmoid(self.z)         # Activation
        self.y = self.h @ self.W_2 + self.b_2  # Second linear
        return self.y
    
    def zero_grad(self):
        ### Each parameter has an additional array associated
        ### with it to store its gradient.  This is not 
        ### automatically cleared, so we have a method to
        ### clear it.
        for param in self.parameters:
            try:
                param.grad.data[:] = 0.0
            except AttributeError:
                pass

x, y = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

#! Perform normalization
x -= x.mean()
x /= x.std()

#! Convert the labels to a one-hot encoding
encoding = np.zeros((len(y), 10))

for row, label in zip(encoding, y):
    row[int(label)] = 1

x = torch.Tensor(x)
y = torch.Tensor(encoding)

#! Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

loss_list = []
net = NeuralNet(784,20,10) # Instantiate network
loss = nn.CrossEntropyLoss()
eta = 1e-1               # Set learning rate (empirically derived)
gamm1 = 1e-1
gamm2 = 1e-2

for t in range(2000):   # run for 50000 epochs
    y_pred = net.forward(x_train)   # Make a prediction
    #L = loss(y_pred,y_train.argmax(-1)) + (gamm1 * torch.sum(net.W_1 ** 2)) + (gamm2 * torch.sum(net.W_2 ** 2)) # Compute mse
    L = loss(y_pred,y_train.argmax(-1)) + (gamm1 * torch.sum(torch.abs(net.W_1))) + (gamm2 * torch.sum(torch.abs(net.W_2))) # Compute mse
    net.zero_grad()           # Clear gradient buffer
    L.backward()              # MAGIC: compute dL/d parameter
    loss_list.append(L.item())
    for param in net.parameters:            # update parameters w/
        param.data -= eta*param.grad.data   # GD
        
    if t%100==0:         # Print loss 
        print(t,L.item())

y_pred = net.forward(x_train)

plt.plot(loss_list[10:])
plt.title('Loss over time')
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(net.W_1[:, 2].reshape(28, 28).detach())
plt.show()