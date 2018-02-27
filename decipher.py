
import nltk
import matplotlib
import random
import re
import sys
import numpy as np
import nltk
from nltk.tag import hmm
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.probability import (FreqDist, ConditionalFreqDist,
                              ConditionalProbDist, DictionaryProbDist,
                              DictionaryConditionalProbDist,
                              LidstoneProbDist, MutableProbDist,
                              MLEProbDist, RandomProbDist)
from sphinx.util.pycompat import sys_encoding
from sqlalchemy.sql.expression import false


path = '/Users/anand/OneDrive/University McGill/NLP/Assignment/Assignment2/Assignment2data'

laplace_mode = False
improved_mode = False
improved_laplace = False

if(len(sys.argv) == 2):
    path = sys.argv[1]
    
if(len(sys.argv) == 3):
    path = sys.argv[2]
    if(sys.argv[1] == '-laplace'):
        laplace_mode = True  
    if(sys.argv[1] == '-lm'):
        improved_mode = True      
if(len(sys.argv) == 4):
    path = sys.argv[3]
    improved_laplace = True
    improved_mode = True
    




C1_test_cipher, C2_test_cipher, C3_test_cipher = [],[],[]
C1_train_cipher, C2_train_cipher, C3_train_cipher = [],[],[]
C1_test_plain, C2_test_plain, C3_test_plain = [],[],[]
C1_train_plain, C2_train_plain, C3_train_plain = [],[],[]

############################################ Reading Data from Files##############################################
for i in range(0,3):
    folder = ''
    training_cipher, training_plain, testing_cipher, testing_plain = [],[],[],[]
    
    if(i == 0):
        folder = '/cipher1/c1_'
    if(i == 1):
        folder = '/cipher2/c2_'
    if(i == 2):
        folder = '/cipher3/c3_'
                
    with open(path + folder + 'train_cipher.txt', 'rb') as cipher:
        for c_text in cipher:
            training_cipher.append(c_text)
    del(cipher,c_text)

    with open(path + folder + 'train_plain.txt', 'rb') as plain: 
        for p_text in plain:
            training_plain.append(p_text)
    del(plain,p_text)

    with open(path + folder + 'test_cipher.txt', 'rb') as cipher:
        for c_text in cipher:
            testing_cipher.append(c_text)
    del(cipher,c_text)

    with open(path + folder + 'test_plain.txt', 'rb') as plain:
        for p_text in plain:
            testing_plain.append(p_text)
    del(plain,p_text)
    
        
    if(i == 0):
        C1_train_cipher = training_cipher
        C1_test_cipher = testing_cipher
        C1_train_plain = training_plain
        C1_test_plain = testing_plain
    if(i == 1):
        C2_train_cipher = training_cipher
        C2_test_cipher = testing_cipher
        C2_train_plain = training_plain
        C2_test_plain = testing_plain
    if(i == 2):
        C3_train_cipher = training_cipher
        C3_test_cipher = testing_cipher
        C3_train_plain = training_plain
        C3_test_plain = testing_plain
                
##################################### Defining Lists of States and Symbols for HMM ###########################################
states = symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ' ']

C1_sequences = []
C2_sequences = []
C3_sequences = []

for i in range(len(C1_train_cipher)):
        C1_sequences.append(zip(C1_train_cipher[i],C1_train_plain[i]))
        
for i in range(len(C2_train_cipher)):
        C2_sequences.append(zip(C2_train_cipher[i],C2_train_plain[i]))
        
for i in range(len(C3_train_cipher)):
        C3_sequences.append(zip(C3_train_cipher[i],C3_train_plain[i]))        
        

trainer = HiddenMarkovModelTrainer(symbols,states)
print("################## Analysis of Ciphers without improved Plaintext modelling ####################### \n")

if(laplace_mode):
    print("################## Laplace ####################### \n")
    C1_tagger = trainer.train_supervised(C1_sequences, estimator= nltk.probability.LaplaceProbDist)
    C2_tagger = trainer.train_supervised(C2_sequences, estimator= nltk.probability.LaplaceProbDist)
    C3_tagger = trainer.train_supervised(C3_sequences, estimator= nltk.probability.LaplaceProbDist)
else:
    C1_tagger = trainer.train_supervised(C1_sequences)
    C2_tagger = trainer.train_supervised(C2_sequences)
    C3_tagger = trainer.train_supervised(C3_sequences)

C1_tester = []
C2_tester = []
C3_tester = []
for i in range(len(C1_test_cipher)):
        C1_tester.append(zip(C1_test_cipher[i],C1_test_plain[i]))
for i in range(len(C2_test_cipher)):
        C2_tester.append(zip(C2_test_cipher[i],C2_test_plain[i]))
for i in range(len(C3_test_cipher)):
        C3_tester.append(zip(C3_test_cipher[i],C3_test_plain[i]))


print("\n ################## C1 Decryption Results #######################")
for row in C1_test_cipher:
    print(C1_tagger.best_path(row))

print("\n ################## C1 Accuracy Results #######################")
print(C1_tagger.test(C1_tester))

print("\n ################## C2 Decryption Results #######################")
for row in C2_test_cipher:
    print(C2_tagger.best_path(row))

print("\n ################## C2 Accuracy Results #######################")
print(C2_tagger.test(C2_tester))

print("\n ################## C3 Decryption Results #######################")
for row in C3_test_cipher:
    print(C3_tagger.best_path(row))

print("\n ################## C3 Accuracy Results #######################")
print(C3_tagger.test(C3_tester))

##################################################### Improved Model ##########################################################################

if(improved_mode):
    print("################## Analysis of Ciphers with Improved Plaintext Modelling ####################### \n")
    X_train = []
    
    with open('C:/Users/anand/Documents/NLPAssignment2/rt-polaritydata/rt-polarity.pos', 'rb') as corpus:
            for sentence in corpus:
                X_train.append(sentence)
    
    with open('C:/Users/anand/Documents/NLPAssignment2/rt-polaritydata/rt-polarity.neg', 'rb') as corpus:
            for sentence in corpus:
                X_train.append(sentence)
    
    for i in range (0,3): 
        if(i == 0):
            train_plain = C1_train_plain
            test_plain = C1_test_plain
            test_cipher = C1_test_cipher
            sequences = C1_sequences
            tester = C1_tester
        if(i == 1):
            train_plain = C2_train_plain
            test_plain = C2_test_plain
            test_cipher = C2_test_cipher
            sequences = C2_sequences
            tester = C2_tester
        if(i == 2):
            train_plain = C3_train_plain
            test_plain = C3_test_plain
            test_cipher = C2_test_cipher
            sequences = C3_sequences 
            tester = C3_tester
                      
        X_train = X_train + train_plain
        
        ####################### Pre-Processing #########################################################
        for i in range(len(X_train)):
            sentence = X_train[i]
            sentence = sentence.lower()
            regex = re.compile('[^a-zA-Z,. ]')
            sentence = regex.sub('',sentence)
            X_train[i] = sentence
        
        starting = FreqDist()
        transitional = ConditionalFreqDist()
        emissional = ConditionalFreqDist()
        Pi = FreqDist()
                   
        for row in test_plain:
            Pi[row[0]] +=1 
                       
        # transition prob
        for row in X_train:
            lasts = None
            for ch in list(row):
                if(lasts is not None):
                    transitional[lasts][ch] += 1
                lasts = ch
         
        # emission prob
        for row in sequences:
            for pair in row:
                emissional[pair[1]][pair[0]] += 1
        
        if(improved_laplace): 
            print("################## Laplace ####################### \n")
            estimator= nltk.probability.LaplaceProbDist 
        else:
            estimator = lambda fdist, bins: MLEProbDist(fdist)
            
        N = len(symbols)
        PI = estimator(Pi, N)
        A = ConditionalProbDist(transitional, estimator, N)
        B = ConditionalProbDist(emissional, estimator ,N)
         
        tagger = HiddenMarkovModelTagger(states, symbols, A, B, PI)
        print("\n ################## C{} Decryption Results #######################".format(int(i)) )
        for row in test_cipher:
            print(tagger.best_path(row))
        
        print("\n ################## C{} Accuracy Results #######################". format(int(i)) )
        print(tagger.test(tester))