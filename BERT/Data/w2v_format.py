
from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

# fin = open(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',"r")

fin = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',header=None,sep="\t")
fin.columns = ['tweet_id', 'topic', 'sentiment_score', 'tweet_text']

fout = open (main_dir+'SemEval2017-task4-dev.subtask-BD.english.w2v.txt',"w")

## format is 
## 681563394940473347      amy schumer     negative        @MargaretsBelly Amy Schumer is the stereotypical 1st world Laci Green feminazi. Plus she's unfunny

for index,line in fin.iterrows(): 
  fout.write( line['tweet_text'] +"\n" ) 


fout.close()




