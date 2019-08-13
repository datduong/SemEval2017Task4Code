

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

import gender_guesser.detector as gender

GenderDetector = gender.Detector(case_sensitive=False)

def GetGender (name): 
  # name = name.split()
  name = word_tokenize(name)
  if len(name) == 1: 
    return GenderDetector.get_gender(name[0]) ## only 1 entry
  # long or weird names 
  gender = " ".join([ GenderDetector.get_gender(n) for n in name ]) ## we don't know how to average this ?? 
  gender = re.sub ("female","FEMALE",gender) ## will make regx easier later
  return gender


main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
os.chdir(out_dir)

"""
1. write out user description 
2. user name --> gender
3. partition data to train w2v
"""


df_user = pd.read_csv(main_dir+'output_semeval_userinfo.tsv',sep="\t")
# fout.write('tweet_id\tuser_id\tfollower_count\tstatus_count\tdescription\tfriend\tcount\tlocation\tlanguage\tname\ttime_zone')
user_info = {} ## name/group/tweet
user_gender = []
for index,row in tqdm( df_user.iterrows() ): 
  if row['user_id'] not in user_info:
    user_info[row['user_id']] = 1
  else: 
    user_info[row['user_id']] = user_info[row['user_id']] + 1 ## count user occurances 
  ## 
  if row['name'] is np.NaN: 
    user_gender.append( 'unknown')
  else:   
    user_gender.append ( GetGender( row['name'] ) ) 


df_user['user_gender'] = user_gender
df_user.to_csv ( "output_semeval_userinfo.gender.tsv" , sep="\t", index=None) 


# print ('\nnum unique user {}'.format(len(user_info)))
# for k in user_info.keys(): 
#   if user_info[k] > 1: 
#     print (k)

# 628281751597645824  1882672394
# 634444835018174464  1882672394

df_tweet = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',header=None,sep="\t")
# fout.write('tweet_id\tuser_id\ttopic\ttext')
df_tweet.columns = ['tweet_id','topic','score','text']

topic_info = {} ## name/group/tweet
topic_gender = []
tweet_text = [] ## we want to format it without any \t\n
for index,row in tqdm( df_tweet.iterrows() ): 
  if row['topic'] not in topic_info:
    topic_info[row['topic']] = 1
  else: 
    topic_info[row['topic']] = topic_info[row['topic']] + 1 ## count user occurances 
  ## 
  topic_gender.append ( GetGender( row['topic'] ) ) 
  tweet_text.append ( row['text'].replace("\n"," ").replace("\t"," ").replace("\r"," ") )
  # if row['tweet_id'] == 636204035910053889:
  #   break 


df_tweet['topic_gender'] = topic_gender
df_tweet['text'] = tweet_text
df_tweet.to_csv ( "SemEval2017-task4-dev.subtask-BD.english.INPUT.gender.tsv" , sep="\t" , index=None) # , index=None

# 636204035910053889

## merge
print ('tweet df count {}'.format(df_tweet.shape) )
print ('user df count {}'.format(df_user.shape) )

df_merge = pd.merge( df_tweet, df_user, on=['tweet_id'] )
print ('merge df count {}'.format(df_merge.shape) )

df_merge.to_csv ( "output_semeval_tweet_userinfo.gender.tsv" , sep="\t", index=None) 

