
## follow the qnli format.
## index, sent1, sent2, some other meta-data ..., label


from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm


def submitJobs (main_dir, output_file, to_skip) :

  if to_skip == "none":
    add_name = ""
    to_skip = []
  else:
    add_name = re.sub(r"\+","_",to_skip)
    to_skip = to_skip.strip().split("+")
    print (to_skip)
    

  # task4B_bert_sentiment_file_notweet
  # main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'

  bert_pretrain_file = open(main_dir+output_file+"_"+add_name+".txt","w", encoding="utf-8" )
  bert_pretrain_file.write("index\tuser_name\tuser_desc\tuser_loc\tuser_gender\ttweet_topic\ttweet_text\ttweet_id\tlabel\n")

  ## read tweet that has user info only ??
  df = pd.read_csv(main_dir+'output_semeval_tweet_userinfo.gender.tsv',sep="\t",dtype=str)
  # tweet_id  topic score text  topic_gender  user_id follower_count  status_count  description friend_count  location  language  name  time_zone user_gender

  tweet_with_user = {}

  counter = 0
  for index,row in tqdm ( df.iterrows() ) :

    if row['tweet_id'] not in tweet_with_user:
      tweet_with_user[row['tweet_id']] = 1 ## just dummy

    label = "entailment"
    if row['score'] == 'negative':
      label = "not_entailment" ## avoid changing BERT code if we do this

    user_text = ""
    for s in ['name','description','location','user_gender']:

      if s in to_skip:
        user_text = user_text +'[MASK]' + "\t" ## add mask to user text because we exclude
        continue

      if row[s] is np.NaN:
        user_text = user_text +'[MASK]' + "\t" ## add mask to user text because no info
      else:
        if s == 'user_gender':
          this_row = re.sub('unknown',' ',row[s]).strip()
          if len(this_row) == 0:
            this_row='[MASK]'
          else:
            this_row = this_row.split()[0].lower().replace("_", " ") ## first occurrance. for example, Mary Jonhson, will be "female male", but we want only the female
        else:
          this_row = row[s].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        ## found some user info, so we add in proper description or name or whatever
        user_text = user_text + this_row + "\t"

    ## complete user text information
    user_text = user_text.strip() ## use strip() to remove last \t  ##.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    if 'text' in to_skip:
      text_sent = user_text + "\t" + row['topic'] + "\t[MASK]\t" + row['tweet_id'] + "\t" + label # " " + row_text
    else:
      row_text = row['text'].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
      text_sent = user_text + "\t" + row['topic'] + "\t" + row_text + "\t" + row['tweet_id'] + "\t" + label # " " + row_text

    bert_pretrain_file.write( str(counter) + "\t" + text_sent + "\n") ## blank between document
    counter = counter + 1


  ### !!!!

  # if 'text' in to_skip:
  #   bert_pretrain_file.close()
  #   print ('here, we ignore the text, so we do not write topic-->score alone ?? exit')
  #   exit() 
    
  ## 

  df = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',header=None,sep="\t",dtype=str)
  df.columns = ['tweet_id', 'topic', 'sentiment_score', 'text']

  for index,row in tqdm ( df.iterrows() ) :

    if row['tweet_id'] in tweet_with_user: ## already seen
      continue

    ## these are tweets without any user at all.

    label = "entailment"
    if row['sentiment_score'] == 'negative':
      label = "not_entailment" ## avoid changing BERT code if we do this

    if 'text' in to_skip:
      text_sent = "[MASK]\t[MASK]\t[MASK]\t[MASK]\t" + row['topic'] + "\t[MASK]\t" + row['tweet_id'] + "\t" + label # " " + row_text
    else: 
      row_text = row['text'].strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
      text_sent = "[MASK]\t[MASK]\t[MASK]\t[MASK]\t" + row['topic'] + "\t" + row_text + "\t" + row['tweet_id'] + "\t" + label # " " + row_text

    bert_pretrain_file.write( str(counter) + "\t" + text_sent + "\n") ## blank between document
    counter = counter + 1


  bert_pretrain_file.close()


if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] )

