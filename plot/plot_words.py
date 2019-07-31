
## import github needed 

import sys,re,os,pickle 

sys.path.append ( "E:/diachronic_embeddings/histwords" )

from representations.embedding import Embedding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it

from viz.scripts import closest_over_time
from viz import mplot, mcommon
from subprocess import check_output

import random
random.seed(1111)


## !!   use python2, because github uses python2 

model_dir = 'C:/Users/dat/Documents/SemEval2017Task4/4B-English/W2vEmbTweetall'
os.chdir(model_dir)

tweet2017all = Embedding.load('W2vEmbTweetall300')

tweet2017all.closest('schumer')

mplot.plot_one_word("schumer", tweet2017all)


## check 



