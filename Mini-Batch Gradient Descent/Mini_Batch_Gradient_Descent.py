import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (9,9)
np.random.seed(0)

# Create constantly-spaced x-values
x = np.linspace(0,1,21)

# Create a linear function of $x$ with slope 1, intercept 1, and normally distributed error with sd=1
y = x + np.random.randn(len(x))*0.1 + 1.0

# Create an initial guess for the weights
w = np.array([0.,0.])

# Define cost function (sum square error between prediction and data points)
def L(w,x,y):
    return 1./2.*sum((y - w[0] - w[1]*x)**2)

# Evaluate the cost function at many values of slope and intercept, aka the brute force method.  
# To be used for visualization purposes
L_grid = np.zeros((101,101))
w0s = np.linspace(0,1.5,101)
w1s = np.linspace(0,1,101)
for i,w1 in enumerate(w1s):
    for j,w0 in enumerate(w0s):
        L_grid[i,j] = L([w0,w1],x,y)

# Create a linear function of $x$ with slope 1, intercept 1, and normally distributed error with sd=1
y = x + np.random.randn(len(x))*0.1 + 1.0

# Gradient function
def G(w,x,y):
    return np.array([-sum(y - w[0] - w[1]*x),-sum((y - w[0] - w[1]*x)*x)])

# Initialize a list to hold our weight values at each step of gradient descent
w_batch = [w.copy()]

# Set the learning rate
eta = 0.01

# Loop over the data 10000 times
for i in range(10000):
    
    # Update the weights by taking a small step in the negative direction of the gradient
    w -= eta*G(w,x,y)
    
    # Append the new parameters to our list of weights
    w_batch.append(w.copy())
    
# Convert the list to a numpy array
w_batch = np.array(w_batch)

# Initialize weights
w = np.array([0.,0.])

# Initialize arrays to hold weights
w_stoch = [w.copy()]
w_epoch = [w.copy()]

# Define learning rates
eta = 0.01
k = 6

# Train for 10000 epochs
for i in range(10000):
    
    # Draw random indices of the dataset
    random_indices = np.random.choice(range(len(x)),len(x),replace=False)

    # Loop over all of the data points without replacement
    for j in range(0, len(random_indices), k):

        mini = random_indices[j:j+k]
        
        # Take as a sample the j-th element in the training data
        x_sample = x[mini]
        y_sample = y[mini]
        
        # Take a gradient descent step based on that single data point
        w -= eta*G(w, x_sample, y_sample)
        w_stoch.append(w.copy())
    
    # Store the weights at the end of the epoch
    w_epoch.append(w.copy())

# Convert lists to arrays
w_stoch = np.array(w_stoch)
w_epoch = np.array(w_epoch)

# Plot the error surface
plt.contour(w0s,w1s,L_grid,200)

# Plot the results of batch gradient descent in green
plt.plot(w_batch[:,0],w_batch[:,1],'go-')

# Plot the results of stochastic gradient descent in blue
plt.plot(w_stoch[:,0],w_stoch[:,1],'b-')
plt.plot(w_epoch[:,0],w_epoch[:,1],'bo')

plt.axis('equal')
plt.xlabel('Intercept')
plt.ylabel('Slope')
plt.show()

plt.plot(w_batch[:,0],w_batch[:,1],'go-')
plt.plot(w_stoch[:,0],w_stoch[:,1],'b-')
plt.plot(w_epoch[:,0],w_epoch[:,1],'bo')

plt.xlim(1.1,1.2)
plt.ylim(0.7,0.8)

plt.xlabel('Intercept')
plt.ylabel('Slope')
plt.show()