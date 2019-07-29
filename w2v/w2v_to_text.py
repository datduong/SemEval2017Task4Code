

import os, sys, re, gensim, pickle
import numpy as np
os.chdir ('/u/scratch/d/datduong/SemEval2017Task4/4B-English/W2vEmbTweet')

## convert w2v in gensim into .txt so that we can train using the embedding later. 
model = gensim.models.Word2Vec.load("W2vEmbTweet")
model.wv.save_word2vec_format("W2vEmbTweet.txt",binary=False)

## convert no numpy as np format like 
# f50-vocab.pkl
# m50-w.npy    

###  USE PYTHON2 BECAUSE OF LEGACY CODE ?? 

num_vocab = len(model.wv.vocab.keys())
dim = 100

np_format = np.zeros((num_vocab,dim))
vocab = []

fin = open ( "W2vEmbTweet.txt",'r' )

counter = -1
for line in fin: 
  if counter == -1:
    counter = 0  
    continue
  ## make numpy 
  line = line.split()
  vocab.append(line[0])
  np_format [counter] = [float(x) for x in line[1::]]
  counter = 1 + counter 


#

np.save ('W2vEmbTweet100-w.npy', np_format)
pickle.dump( vocab, open("W2vEmbTweet100-vocab.pkl","wb"), protocol=2 ) ## to make it okay with python2.7

print (np_format)

fin.close() 
