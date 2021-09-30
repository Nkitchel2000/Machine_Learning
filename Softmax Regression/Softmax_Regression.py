import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()

#Seperate Dictionary
x = digits["data"]
y = digits["target"]

#! Perform normalization
x -= x.mean()
x /= x.std()

#! Convert the labels to a one-hot encoding
encoding = np.zeros((len(y), 10))

for row, label in zip(encoding, y):
    row[label] = 1

#! Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, encoding, test_size=0.1, random_state=42)

#Softmax funciton to return guess of prediction
def softmax(phi, w):
    phi_weight = np.dot(phi, w)

    #E calculation to make probability of one



