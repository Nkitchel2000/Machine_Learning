import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn

x, y = sklearn.datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

#! Perform normalization
x -= x.mean()
x /= x.std()

#! Convert the labels to a one-hot encoding
encoding = np.zeros((len(y), 10))

for row, label in zip(encoding, y):
    row[int(label)] = 1

#! Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, encoding, test_size=0.1, random_state=42)

#Edit x_train to add a column of ones at the beginning
column_ones = np.ones((len(x_train),1))
x_train = np.hstack((column_ones, x_train))

#Create random values in array for logits
w = np.random.random((785, 10)) * 1e-6

#Softmax function to return guess of prediction
def softmax(phi, w):
    logit = np.dot(phi, w)

    top = np.exp(logit)
    bottom = np.sum(np.exp(logit), axis=1)
    return (top / bottom.reshape(-1 ,1))

#Compute the cost function with three parameters
def L(w, phi, y_obs):

    #CEE matrix
    CEE = np.zeros((len(y_obs), 1))

    #Seperate sections of the L equation
    one_mN = -1 / (len(phi) * 10)
    log_val = np.log(softmax(phi, w))

    for i in range(len(y_obs)):
        CEE[i] = np.dot(y_obs[i], log_val[i].transpose())

    return one_mN * CEE

def grad(w, phi, y_obs):

    #Seperate sections of the grad equation
    one_mN = -1 / (len(phi) * 10)
    soft = softmax(phi, w)

    inside = np.transpose(y_obs - soft)

    return one_mN * np.transpose(inside @ phi)

def grad_descent(w, phi, y_obs, gamma, tol, stop):

    total_tol = []

    while tol > stop:
        w_update = w + (gamma * grad(w, phi, y_obs))
        w_current = w_update
        tol = np.linalg.norm(np.abs(w_update - w_current))
        total_tol.append(tol)

    return w_current, total_tol

final_w, tolerance = grad_descent(w, x_train, y_train, 1e-1, 100, 1e-3)

fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(final_w[1:, 0].reshape(28, 28))
plt.show()