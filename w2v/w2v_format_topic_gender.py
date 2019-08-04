
from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

CHOICE = 'FEMALE'
main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

os.chdir(out_dir)

# fin = open(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',"r")

fout = open (main_dir+'SemEval2017-task4-dev.subtask-BD.english.'+CHOICE+'.w2v.txt',"w")

## append user data to the w2v
user_desc = pd.read_csv ('output_semeval_userinfo.gender.tsv', sep="\t",dtype='str')
# tweet_id  user_id follower_count  status_count  description friend_count  location  language  name  time_zone user_gender
user_id = {} # not write the same user 
for index,line in user_desc.iterrows(): 
  if line['user_id'] in user_id: 
    continue
  user_id[line['user_id']] = 1
  line_out = word_tokenize( " ".join( line[s] for s in ['description','name','location'] if line[s] is not np.NaN) )
  fout.write( " ".join(line_out).lower() +"\n" ) 


fin = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.gender.tsv',sep="\t")
# fin.columns = ['tweet_id', 'topic', 'sentiment_score', 'tweet_text']
## format is 
## 681563394940473347      amy schumer     negative        @MargaretsBelly Amy Schumer is the stereotypical 1st world Laci Green feminazi. Plus she's unfunny
for index,line in fin.iterrows(): 
  gender = line['topic_gender'].strip().split()
  if CHOICE != "all" : ## not picking male/female or some other phrases 
    counter = 1.0 * len ([k for k in gender if CHOICE in k]) ## how many times word like 'male' appear 
    if counter > 0: # / len(gender) >=.33: 
      line_out = word_tokenize( line['text'].lower() ) ## use lower case because too many low freq. words ? 
      ## format topic name so that we encoder whole entity ? 
      line_out = " ".join(line_out) +"\n" 
      topic = re.sub(" ","_",line['topic'].lower())
      line_out = re.sub( line['topic'], topic, line_out )
      fout.write( line_out.strip() +"\n" ) 
  else: 
    line_out = word_tokenize( line['text'].lower() ) ## use lower case because too many low freq. words ? 
    ## format topic name so that we encoder whole entity ? 
    line_out = " ".join(line_out) +"\n" 
    topic = re.sub(" ","_",line['topic'].lower())
    line_out = re.sub( line['topic'], topic, line_out )
    fout.write( line_out.strip() +"\n" ) 



fout.close()


## create male/female partition. 
## create partition by region ? 


