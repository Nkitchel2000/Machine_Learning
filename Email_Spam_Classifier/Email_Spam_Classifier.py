import numpy as np

# Maps from 'ham' or 'spam' strings to zero or one
def mapper(s):
    if s=='spam':
        return 0
    else:
        return 1

# Read in the text file
f = open('SMSSpamCollection.txt','r')
lines = f.readlines()

# Break out the test data
test_lines = lines[:len(lines)//5]
lines = lines[len(lines)//5:]

# Instantiate the frequency dictionary and an array to
# record whether the line is ham or spam
word_dictionary = {}
training_labels = np.zeros(len(lines),dtype=int)

ham_count = 0
spam_count = 0
total = len(lines)
# Loop over all the training messages
for i,l in enumerate(lines):
    #count total
    # Split into words
    l = l.lower().split()
    # Record the special first word which always ham or spam
    if l[0]=='ham':
        ham_count = ham_count + 1
        training_labels[i] = 1
    # For each word in the message, record whether the message was ham or spam
    for w in l[1:]:
        # If we've never seen the word before, add a new dictionary entry
        if w not in word_dictionary:
            word_dictionary[w] = [1,1]
        word_dictionary[w][mapper(l[0])] += 1
        
# Loop over the test messages
test_labels = np.zeros(len(test_lines),dtype=int)
test_messages = []
for i,l in enumerate(test_lines):
    l = l.lower().split()
    if l[0]=='ham':
        test_labels[i] = 1
    test_messages.append(l)

counts = np.array([v for v in word_dictionary.values()]).sum(axis=0)
spam_count = total - ham_count

#What is the prior P(Y=ham) ?
ham_prior = ham_count/total
spam_prior = spam_count/total

## What are the class probabilities P(X=word|Y=ham) for each word?
ham_likelihood = {}
spam_likelihood = {}
#Spam is the first column in the dictionary, ham is second column
for key,val in word_dictionary.items():
   ham_likelihood[key] =   word_dictionary[key][1]/counts[1]
   spam_likelihood[key] = word_dictionary[key][0]/counts[0]

# Where to hold the ham and spam posteriors
posteriors = np.zeros((len(test_lines),2))

# Loop over all the messages in the test set
for i,m in enumerate(test_messages):
    posterior_ham = 1.0
    posterior_spam = 1.0
    #Don't forget to include the prior!
    posterior_ham *= ham_prior
    posterior_spam *= spam_prior
    
    # Loop over all the words in each message
    for w in m:
        #What is the purpose of this try/except handler?
        #The try is here to allow for the case where a word doesn't exist in either ham or spam emails, allowing the loop to continue regardless
        try:
            posterior_ham *= (ham_likelihood[w])
            posterior_spam *= (spam_likelihood[w])
        except KeyError:
            pass
    
    # Notice the normalization factor (denominator) 
    # to turn these into proper probabilities!
    posteriors[i,0] = posterior_spam/(posterior_spam + posterior_ham)
    posteriors[i,1] = posterior_ham/(posterior_spam + posterior_ham)

#Use the argmax function to compute the highest value in the line that is created in the posteriors varibles
