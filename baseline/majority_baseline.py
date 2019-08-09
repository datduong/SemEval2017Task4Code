

import re,sys,os,pickle,gzip
import numpy as np
import pandas as pd

## for each topic, count fraction positive/negative


score = {} ## 'topoic' [negative positive]
fin = open("","r")
for line in fin:
  line = line.strip().split("\t")
  if line[1] in score:
    score[line[1]] = [line[2]]
  else:
    score[line[1]].append(line[2]) ## append score

#
fin.close()

# count how many positive/negative
for topic in score:
  negative = score[topic].count('negative')
  positive = score[topic].count('positive')
  this_score = np.array ([negative,positive])
  this_score = this_score / np.sum(this_score)
  score[topic] = this_score


## 
pickle.dump(score,open("majority_score_by_topic.pickle","rb"))









