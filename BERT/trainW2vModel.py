
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
  def __init__(self, dirname):
    self.dirname = dirname
  def __iter__(self):
    for fname in os.listdir(self.dirname):
      if fname != "SemEval2017-task4-dev.subtask-BD.english.w2v.txt":
        continue
      print (fname)
      for line in open(os.path.join(self.dirname, fname)):
        yield [x.strip() for x in line.split()]


def submitJobs (path2TextFiles , file2savePath, modelName2save, dimensionOfVec):
  print ('save path here {}'.format(file2savePath))
  if not os.path.exists(file2savePath):
    os.mkdir(file2savePath)
  ###
  print ("begin\n")
  print ("now loading sentences\n")
  sentences = MySentences(path2TextFiles)
  print ('now running model\n')
  ## using min_count = 0 to keep all the words appearing in tweet. didn't seem to be that many words
  model = gensim.models.Word2Vec(sentences,min_count=0,size=dimensionOfVec,max_vocab_size=1500000,workers=4,window=5)
  print ('finished running, now save file\n')
  model.save(os.path.join(file2savePath,modelName2save))
  print ('finished saving file\n')


### -------------------------------------------------------

if len(sys.argv)<1:
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) )





