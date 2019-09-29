

from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/TweetShootData2018/'
out_dir = '/u/scratch/d/datduong/TweetShootData2018'

os.chdir(main_dir)

## create pretrain data for Mask LM 

list_name = """nashville
pittsburgh
santa_fe
thousand_oaks
dallas
colorado_springs
chattanooga
burlington
baton_rouge
fresno
fort_lauderdale
roseburg
parkland
orlando
kalamazoo
sutherland_springs
san_francisco
san_bernardino
vegas
thornton
annapolis
"""

np.random.seed(1234)

list_name = list_name.split() 

# fout = open ("tweet_pretrain_25percent.txt","w")

fout1 = open('tweet_pretrain_25percent_train.txt','w')
fout2 = open('tweet_pretrain_25percent_test.txt','w')


for add_on in list_name :

  print ('name {}'.format(add_on))
  df = pd.read_csv ( "user_data_with_tweet_"+add_on+".tsv", sep="\t", dtype=str) # index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

  for counter,row in df.iterrows(): 

    if np.random.uniform() > .25: 
      continue

    try: 
      ## may have NaN ?? just skip this sample
      str1 = " ".join (row[s].strip() for s in ['user_name', 'user_desc', 'user_loc', 'user_gender'] if row[s].strip()!='[MASK]' )
    except: 
      continue
    tweet_text = row['tweet_text'].strip() ## must split into sentences ?? or just split half 
    # text_sent = sent_tokenize(tweet_text) ## array of sent
    # text_sent[0] = row['tweet_topic'] + " " + text_sent[0] ## add topic to 1st sent ? 
    # str1 = str1 + "\n" + "\n".join(sent for sent in text_sent if len(sent)>0)+ "\n\n" ## skip line 
    str1 = str1 + " " + tweet_text + "\n\n" ## skip line , but not break between user_info and tweet_text 

    str1 = re.sub(r'http\S+', ' ', str1) ## remove http

    if np.random.uniform() > .10: 
      fout1.write(str1)
    else: 
      fout2.write(str1)


## end 
fout1.close() 
fout2.close() 

