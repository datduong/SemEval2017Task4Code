from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

fin = open(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.txt',"r")
fout = open (main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',"w")

for line in fin: 
  fout.write( line.strip()+"\n" )


fout.close()
fin.close() 


