

import os, sys, re, gensim, pickle
import numpy as np
from sklearn import preprocessing



num_vocab = 52131 # 52131 768
dim = 768

np_format = np.zeros((num_vocab,dim))
vocab = []

os.chdir("/local/datdb/SemEval2017Task4/4B-English/BertSentiment/fold_1")
fin = open ( 'word_vector.txt','r', encoding='utf-8' )
which_file = 'word_vector'

keep1 = None
keep2 = None

counter = -1
for line in fin: 
  if counter == -1:
    counter = 0  
    continue
  ## make numpy 
  line = line.split()
  word = line[0:(len(line)-dim)]
  word = "_".join(word) ## make life easier if we don't have space for given word
  vocab.append(word)
  np_format [counter] = [float(x) for x in line[(len(line)-dim):len(line)]]
  #
  if word == 'he': 
    keep1 = np_format [counter]
  if word == 'she': 
    keep2 = np_format [counter]
  #
  counter = 1 + counter 


# check he v.s she distance
keep1 = preprocessing.normalize([keep1])[0]
keep2 = preprocessing.normalize([keep2])[0]
print ( 'see cosine distance of he vs. she , {}'.format(keep1.dot(keep2)) )


np.save (which_file+str(dim)+'-w.npy', np_format)
pickle.dump( vocab, open(which_file+str(dim)+"-vocab.pkl","wb"), protocol=2 ) ## to make it okay with python2.7

print (np_format)

fin.close() 

