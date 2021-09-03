from matplotlib import pyplot as plt
import numpy as np
from numpy import arange

#Function to determine if a person have covid considering they have tested positive for the desease
def having_covid():
    #Probability of Covid in general
    #P(covid) = 7/1000
    prior = 7/1000

    #Probability of showing positive with having Covid
    #P(posivite | covid) = 70%
    likelihood = 7/10

    #P(negative | no Covid)  = 98%
    Pspecificity = 0.02

    #P(positive) is a bunch of stuff to find the varible
    posterior = likelihood * prior + Pspecificity * 993/1000

    return likelihood * prior / posterior

#Function plots the specificity over the probability of having covid
def plot_specificity():
    #Same values from the above function needed in this function
    prior = 7/1000
    likelihood = 7/10

    #Vector of each precentage out of 100 as a decimal
    specificity_vec = arange(0, 1, .01)
    #Calculate the positive percentage of each value into a vector
    positive_vec = likelihood * prior + (1- specificity_vec) * (1- prior)
    #The changes of actually having covid into a percentage
    actually_have_covid = likelihood * prior / positive_vec

    #Plot the graph with labels
    plt.plot(specificity_vec, actually_have_covid)
    plt.ylabel('Probability')
    plt.xlabel('Specificity')
    plt.show()


#Run function
plot_specificity()
