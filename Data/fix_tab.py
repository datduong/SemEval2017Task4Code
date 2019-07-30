from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm


main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

fin = open(main_dir+'output_semeval_userinfo.txt',"r", encoding='utf-8')
fout = open (main_dir+'output_semeval_userinfo.tsv',"w", encoding='utf-8')

fout.write('tweet_id\tuser_id\tfollower_count\tstatus_count\tdescription\tfriend_count\tlocation\tlanguage\tname\ttime_zone')

for line in fin: 
  if "Not Available" in line: 
    continue
  if len(line) == 0:
    continue
  fout.write( '\n'+line.strip() )


fout.close()
fin.close() 


