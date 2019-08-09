
## follow the qnli format.
## index, sent1, sent2, some other meta-data ..., label


from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'


bert_pretrain_file = open(out_dir+"task4B_bert_sentiment_file_notweet.txt","w", encoding="utf-8" )
bert_pretrain_file.write("index\tquestion\tsentence\tlabel\n")

## read tweet that has user info only ??
df = pd.read_csv(main_dir+'output_semeval_tweet_userinfo.gender.tsv',sep="\t")
# tweet_id  topic score text  topic_gender  user_id follower_count  status_count  description friend_count  location  language  name  time_zone user_gender

tweet_with_user = {}

counter = 0
for index,row in tqdm ( df.iterrows() ) :

  if row['tweet_id'] not in tweet_with_user:
    tweet_with_user[row['tweet_id']] = 1 ## just dummy

  label = "entailment"
  if row['score'] == 'negative':
    label = "not_entailment" ## avoid changing BERT code if we do this

  user_text = " ".join( row[s] for s in ['description','name','location'] if row[s] is not np.NaN)
  user_text = user_text.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

  # row_text = row['text'].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

  if len(user_text) > 0 : ## has something
    text_sent = user_text + "\t" + row['topic'] + "\t" + label # " " + row_text

  else:
    text_sent = row['topic'] + "\t" + label ## ignore user name # + row_text + "\t"

  bert_pretrain_file.write( str(counter) + "\t" + text_sent + "\n") ## blank between document
  counter = counter + 1


### !!!!

df = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',header=None,sep="\t")
df.columns = ['tweet_id', 'topic', 'sentiment_score', 'tweet_text']

for index,row in tqdm ( df.iterrows() ) :

  if row['tweet_id'] in tweet_with_user: ## already seen
    continue

  label = "entailment"
  if row['sentiment_score'] == 'negative':
    label = "not_entailment" ## avoid changing BERT code if we do this

  row_text = row['tweet_text'].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

  text_sent = row['topic'] + "\t" + row_text + "\t" + label ## ignore user name row_text + "\t" 

  bert_pretrain_file.write( str(counter) + "\t" + text_sent + "\n") ## blank between document
  counter = counter + 1


bert_pretrain_file.close()

