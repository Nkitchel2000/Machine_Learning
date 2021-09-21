import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('crickets.txt')
data -= data.mean(axis=0)
data /= data.std(axis=0)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from scipy.linalg import pinv

X_train, X_test, Y_train, Y_test = train_test_split(data[:,0], data[:,1], test_size=0.5, random_state=42)

def calc_phi(train, degree):
    matrix = np.array([[0.0 for i in range(degree)] for i in range(len(train))])


    for i in range(len(train)):
        for d in range(degree):

            spot = train[i] ** d

            matrix[i][d] = spot

    return matrix
        

def fit_polynomial(X,Y,d,l):
    """  Find the ordinary least squares fit of an independent 
        variable X to a dependent variable y"""

    lamtrix = np.array([[l for i in range(d)] for i in range(d)])

    lamtrix = l * np.identity(d)

    Phi = calc_phi(X,d)
    PhiT = Phi.transpose()
    yside = np.dot(PhiT, Y)

    lside = (np.dot(PhiT, Phi) + lamtrix)

    w = np.linalg.solve(lside, yside)
    return w

degrees = np.linspace(1,15,15).astype(int)

train_rmse = []
test_rmse = []
for d in degrees:
    #! Use the function you generated above to fit 
    #! a polynomial of degree d to the cricket data
 
    #! Compute and record RMSE for both the training and
    #! test sets.  IMPORTANT: Don't fit a new set of 
    #! weights to the test set!!!

    train_line = fit_polynomial(X_train,Y_train,d, 0)

    train_prediction = np.dot(calc_phi(X_train, d), train_line)
    test_prediction = np.dot(calc_phi(X_test, d), train_line)

    train_rmse.append(np.sqrt(sum((train_prediction - Y_train) ** 2)))
    test_rmse.append(np.sqrt(sum((test_prediction - Y_test) ** 2)))

plt.semilogy(degrees,train_rmse)
plt.semilogy(degrees,test_rmse)
plt.show()

train_rmse = []
test_rmse = []
lamdas = np.logspace(-9,2,12)

d = 15
for lamda in lamdas:
    #! Use the function you generated above to fit 
    #! a polynomial of degree 15 to the cricket data
    #! with varying lambda 
    
    #! Compute and record RMSE for both the training and
    #! test sets.  IMPORTANT: Don't fit a new set of 
    #! weights to the test set!!!

    train_line = fit_polynomial(X_train,Y_train,d, lamda)

    train_prediction = np.dot(calc_phi(X_train, d), train_line)
    test_prediction = np.dot(calc_phi(X_test, d), train_line)

    train_rmse.append(np.sqrt(sum((train_prediction - Y_train) ** 2)))
    test_rmse.append(np.sqrt(sum((test_prediction - Y_test) ** 2)))

plt.loglog(lamdas,train_rmse)
plt.loglog(lamdas,test_rmse)
plt.show()