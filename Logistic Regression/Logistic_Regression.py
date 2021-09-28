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

    return (p1 + p2) - p3

# Develop a method to find the best values of w_0 and w_1.
w_options = [(w0, w1) for w0 in np.arange(-4, 4 , 0.1) for w1 in np.arange(-4, 4, 0.1)] 
store = np.zeros(len(w_options))

for i, k in enumerate(w_options):
    store[i] = L(np.array(w_options[i]), X, Y , 1)

best = w_options[np.argmax(store)]
X_data = np.arange(-3, 3, 0.1)
Y_data = sigmoid((X_data * best[1]) + best[0])

plt.plot(X_data, Y_data)
plt.show()

#! Plot lobster survival
def gradient(w, X, Y, lamda = 0.1):

    phi = np.column_stack((np.repeat(1, len(X)), X))
    log_posterior_gradient = 0.0
    for x, y, phi_i in zip(X, Y, phi):
        log_posterior_gradient += (y - sigmoid(phi_i @ w)) * phi_i
    
    return log_posterior_gradient - lamda * w.T 

gradient(w = np.array(w_options[0]), X = X, Y = Y)

#! use plt.quiver to visualize the gradients of the log-posterior.
log_posterior_gradient_values = []
for i, w in enumerate(w_options):
    log_posterior_gradient_values.append(gradient(w = np.array(w_options[i]), X = X, Y = Y))

length = int(len(w_options) ** 0.5)

W0_options = np.arange(len(w_options))
W1_options = np.arange(len(w_options))
W0_gradient = np.zeros(len(w_options))
W1_gradient = np.zeros(len(w_options))

index = range(len(w_options))

for i in index:
    W0_options[i] = w_options[i][0]
    W1_options[i] = w_options[i][1]
    W0_gradient[i] = log_posterior_gradient_values[i][0]
    W1_gradient[i] = log_posterior_gradient_values[i][1]

u = W0_gradient.reshape(length, length)
v = W1_gradient.reshape(length, length)

length = int(len(w_options) ** 0.5)
log_posterior_gradient = store.reshape( (length, length) )

plt.quiver(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1), u, v, scale = 5000)
plt.show()