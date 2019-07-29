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
fout = open (main_dir+'output_5th_semeval_userinfo.tsv',"w")

fout.write('tweet_id\tuser_id\tfollower_count\tstatus_count\tdescription\tfriend\tcount\tlocation\tlanguage\tname\ttime_zone\n')

for line in fin: 
  if "Not Available" in line: 
    continue
  fout.write( line.strip()+"\n" )


fout.close()
fin.close() 


