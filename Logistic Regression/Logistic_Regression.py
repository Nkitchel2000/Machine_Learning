import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('datasets/lobster_survive.dat',skiprows=1)
X,Y = data.T
X -= X.mean()
X /= X.std()

bins = np.unique(X)
plt.hist(X[Y==0],bins,histtype='step',density=True,label='Died',color='blue')
plt.hist(X[Y==1],bins,histtype='step',density=True,label='Survived',color='red')
plt.legend()
plt.show()

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def L(w,X, Y, lam):
    
    #Add a column of 1's into X so the dementions match when dot producting with w
    newX = [ [ 0 for i in range(2) ] for j in range(len(X)) ]
    for i in range(len(X)):
        newX[i][1] = X[i]

    #Transposes W
    wT = np.transpose(w)

    #This is our matrix that is passed into sigmoid
    z = np.dot(newX, w)
    wX = sigmoid(z)

    #Parts of the L equation, all added together at the bottom to comput the log postieror
    p1 = np.dot(Y, wX)
    p2 = np.dot(1 - Y,np.log(1 - wX))
    p3 = ( lam * np.dot(wT, w))

    #The sum is computed inside each dot product as it multiples each index by the others and adds them up.
    log_post = (p1 + p2) - p3

    return log_post


# Develop a method to find the best values of w_0 and w_1.
w = [[1], [2]]
print(L(w, X, Y, 3)[0][0])