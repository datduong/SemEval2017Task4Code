

## need to check user info for completeness, download from twitter breaks, so have to restart several times. 

""" 
concat all the tweeter download files first 
cat ... 
"""

from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm


main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

fin = open(main_dir+'output5.txt_semeval_userinfo.txt',"r")

## replace \n with nothing 
line_so_far = ""
for line in fin: 
  line = line.strip()
  if len(line) == 0: 
    line_so_far = line_so_far + line

