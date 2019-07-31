
import gensim, logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import stat
import fileinput
import time
import random
import sys, traceback
import subprocess
from subprocess import Popen, PIPE
import re
import gzip


class MySentences(object):

  def __init__(self, dirname, file_wanted=None):
    self.dirname = dirname
    self.file_wanted = file_wanted

  def __iter__(self):

    for fname in os.listdir(self.dirname):

      if self.file_wanted is not None: ## we can do this better, but ehhhh... whatever
        if fname != self.file_wanted:
          continue

      print (fname)

      for line in open(os.path.join(self.dirname, fname)):
        ## use lower case ??
        yield [x.strip() for x in line.split()]


def submitJobs (path2TextFiles , file2savePath, file_wanted, modelName2save, dimensionOfVec):

  print ('save path here {}'.format(file2savePath))

  if not os.path.exists(file2savePath):
    os.mkdir(file2savePath)

  print ("now loading sentences\n")
  sentences = MySentences(path2TextFiles,file_wanted)

  print ('now running model\n')
  ## using min_count = 0 to keep all the words appearing in tweet. didn't seem to be that many words
  model = gensim.models.Word2Vec(sentences,min_count=0,size=dimensionOfVec,max_vocab_size=150000,workers=4,window=5,iter=100)

  print ('finished running, now save file\n')
  model.save(os.path.join(file2savePath,modelName2save))
  print ('finished saving file\n')


### -------------------------------------------------------

if len(sys.argv)<1:
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) )





