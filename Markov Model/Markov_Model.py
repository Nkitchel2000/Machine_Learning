import pickle
import numpy as np

# Define some useful constants
N_nucleobases = 4
N_classes = 2
nucleobases = ['A','T','G','C']

# Load the training data using pickle
sequences,labels = pickle.load(open('gene_training_unix.p','rb'))

# Initialize the class priors and transition matrices
pi_0 = np.zeros((N_nucleobases))
pi_1 = np.zeros((N_nucleobases))
pic_0 = 0; pic_1 = 0

nucleobase_count_0 = [0,0,0,0]
nucleobase_count_1 = [0,0,0,0]
total0 = 0; total1 = 0

A_0 = np.zeros((N_nucleobases,N_nucleobases))
A_1 = np.zeros((N_nucleobases,N_nucleobases))

##### Train prior #####
#! Compute unconditional nucleobase probabilities

for s, l in zip(sequences, labels):
    if l == 0:
        pic_0 += 1
        total0 += 20
        for letter in s:
            if letter == nucleobases[0]:
                nucleobase_count_0[0] += 1
            if letter == nucleobases[1]:
                nucleobase_count_0[1] += 1
            if letter == nucleobases[2]:
                nucleobase_count_0[2] += 1
            if letter == nucleobases[3]:
                nucleobase_count_0[3] += 1
    if l == 1:
        pic_1 += 1
        total1 += 20
        for letter in s:
            if letter == nucleobases[0]:
                nucleobase_count_1[0] += 1
            if letter == nucleobases[1]:
                nucleobase_count_1[1] += 1
            if letter == nucleobases[2]:
                nucleobase_count_1[2] += 1
            if letter == nucleobases[3]:
                nucleobase_count_1[3] += 1

# Convert from counts to probabilities by normalizing
pic_0 /= (total0 + total1)
pic_1 /= (total0 + total1)

for i in range(4):
    pi_0[i] = (nucleobase_count_0[i] / total0)

for i in range(4):
    pi_1[i] = (nucleobase_count_1[i] / total1)

options_of_nucleobases = ["AA", "AT", "AC", "AG",
                         "TA", "TT", "TC", "TG",
                         "CA", "CT", "CC", "CG",
                         "GA", "GT", "GC", "GG"]

##### Train transition matrix #####
for s,l in zip(sequences,labels):
    sequence_length = len(s)
    for p in range(sequence_length-1):
        #! s is a length 20 sequence of nucleoboases, for all s, count the number of times that a nucleobase 
        #! transitions to another nucleobase and record this information in the appropriate transition matrix (A_0 or A_1)

        if l == 0:
            #Means the label is 0 so will be using the A_0 matrix
            for i in range(16):
                if s[p:p + 2] == options_of_nucleobases[i]:
                    A_0[int((i - (i % 4)) / 4)][i % 4] += 1
        if l == 1:
            #Means the label is 1 so will be using the A_1 matrix
            for i in range(16):
                if s[p:p + 2] == options_of_nucleobases[i]:
                    A_1[int((i - (i % 4)) / 4)][i % 4] += 1

        pass

# Convert from counts to probabilities by row normalization
A_0/=A_0.sum(axis=1)[:,np.newaxis]
A_1/=A_1.sum(axis=1)[:,np.newaxis]
sum0 = 1
sum1 = 1

##### Generate a synthetic sequence #####
def generate_new_sequence(A,pi,n=20):
    """  Arguments: A -> Nucleobase transition matrix
                    pi -> Prior
                    n -> length of sequence to generate """
    # Draw from the prior for the first nucleobase
    s = [np.random.choice(nucleobases,p = pi)]
    transform = s[0]
    sum = 1

    #! Write the code that uses the transition matrix to produce a length n sample
    A_index = { "A": 0, "T": 1, "C": 2, "G": 3 }
    generated_sequence = ""
    previous = s[0]

    for x in range(20):
        if previous == "A":
            s  = [np.random.choice(nucleobases,p = A_0[A_index["A"]])]
        if previous == "T":
            s  = [np.random.choice(nucleobases,p = A_0[A_index["T"]])]
        if previous == "C":
            s  = [np.random.choice(nucleobases,p = A_0[A_index["C"]])]
        if previous == "G":
            s  = [np.random.choice(nucleobases,p = A_0[A_index["G"]])]

    return [generated_sequence, sum]

sum0 = generate_new_sequence(A_0, pi_0, 20)[1]
sum1 = generate_new_sequence(A_1, pi_1, 20)[1]

sequences_test,labels_test = pickle.load(open('genes_test_unix.p','rb'))
sequence_probabilities_0 = 0; sequence_probabilities_1 = 0
true_prob_0 = 0; true_prob_1 = 0

for s, l in zip(sequences_test, labels_test):
    #! Write a function that evaluates the probability of class membership for each class by multiplying the 
    #! prior by the likelihood over all symbol transitions  
    sum0 = 1; sum1 = 1 
    
    #True values of labels
    if l == 0:
        true_prob_0 += 1
    if l == 1:
        true_prob_1 += 1

    for x in range(20):
        if x > 0:
            final_trans = s[x - 1] + s[x]

            for i in range(16):
                if final_trans == options_of_nucleobases[i]:
                    sum0 *= A_0[int((i - (i % 4)) / 4)][i % 4]
                    sum1 *= A_1[int((i - (i % 4)) / 4)][i % 4]
        else:
            #s[0]
            for i in range(4):
                if s[0] == nucleobases[i]:
                    sum0 *= pi_0[i]
                    sum1 *= pi_1[i]

    class0 = sum0/(sum0 + sum1)
    class1 = sum1/(sum0 + sum1)

    if class0 > class1:
        sequence_probabilities_0 += 1
    if class1 > class0:
        sequence_probabilities_1 += 1

accuracy_0 = 1 - (((true_prob_0 * (sequence_probabilities_0/true_prob_0)) - true_prob_0)/(true_prob_0 + true_prob_1))
accuracy_1 = 1 - ((true_prob_1 - (true_prob_1 * (sequence_probabilities_1/true_prob_1)))/(true_prob_0 + true_prob_1))