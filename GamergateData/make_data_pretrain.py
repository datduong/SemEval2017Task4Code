

from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'


## create pretrain data for Mask LM 

fout = open ("/u/scratch/d/datduong/GamergateTweet/gamergate_finetune_small.txt","w")

df = pd.read_csv ( "/u/scratch/d/datduong/GamergateTweet/SplitData/NotMask/train.tsv", sep="\t", dtype=str) # index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label
for counter,row in df.iterrows(): 
  try: 
    ## may have NaN ?? just skip this sample
    str1 = " ".join (row[s].strip() for s in ['user_name', 'user_desc', 'user_loc', 'user_gender'] if row[s].strip()!='[MASK]' )
  except: 
    continue
  tweet_text = row['tweet_text'].strip() ## must split into sentences ?? or just split half 
  text_sent = sent_tokenize(tweet_text) ## array of sent
  text_sent[0] = row['tweet_topic'] + " " + text_sent[0] ## add topic to 1st sent ? 
  str1 = str1 + "\n" + "\n".join(text_sent)+ "\n\n" ## skip line 
  fout.write(str1)


fout.close() 

