

import os, sys, re, gensim, pickle
import numpy as np
from sklearn import preprocessing


# which_file = 'W2vEmbTweetall'
# dim = 300

def submitJobs (which_file, dim ): 

  os.chdir ('/u/scratch/d/datduong/SemEval2017Task4/4B-English/' + which_file)

  ## convert w2v in gensim into .txt so that we can train using the embedding later. 
  model = gensim.models.Word2Vec.load(which_file)
  model.wv.save_word2vec_format(which_file+".txt",binary=False)

  ## convert no numpy as np format like 
  # f50-vocab.pkl
  # m50-w.npy    

  ###  USE PYTHON2 BECAUSE OF LEGACY CODE ?? ... let's stick with py3

  num_vocab = len(model.wv.vocab.keys())

  np_format = np.zeros((num_vocab,dim))
  vocab = []

  fin = open ( which_file+".txt",'r', encoding='utf-8' )

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


### -------------------------------------------------------

if len(sys.argv)<1:
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], int(sys.argv[2]) )




