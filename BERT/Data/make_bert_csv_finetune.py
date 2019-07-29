
# /local/jyzhao/Github/data/ugb/mult09/tuples/
# https://github.com/JieyuZhao/UGB#dataset

# user_id, restaurant_id, sentiment_score, concept_ids, review_id, review_text

# example ...
# 1337    10400   4       1 67 73 76 14 15 48 88  MN4muG24xx58_Qt8kxayHw  This place is always so packed! We made reservations for 11 of us at 7pm last night but got there around 5:30 to try and get patio seating ...


from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize

import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/YelpReviewUserEmb/Data/TextData/'
out_dir = '/u/scratch/d/datduong/YelpReviewUserEmb/Data/TextData/'

user_map = {} ## map user {1:someID}
df = pd.read_csv(main_dir+'yelp_recursive.usermap',header=None,sep="=")
user = list(df[0])
userid = list (df[1])
for i in range(len(user)):
  user_map[user[i]]=userid[i]


vocab = {} ## also count, do this for ALL test/train/dev ?

## do we need BERT for pretrained ? # want: one sentence per line, and one blank line between documents.
## we need BERT for sentiment score

for data_type in ['dev','test','train']:

  df = pd.read_csv(main_dir+data_type+'.txt',header=None,sep="\t")
  df.columns = ['user_id', 'restaurant_id', 'sentiment_score', 'concept_ids', 'review_id', 'review_text']

  bert_pretrain_file = open(out_dir+data_type+"_bert_pretrain_file.txt","w")
  bert_sentiment_file = open(out_dir+data_type+"_bert_sentiment_file.txt","w")

  for index,row in tqdm ( df.iterrows() ) :

    text = row['review_text'] ##.lower()

    word = word_tokenize(text) ## count word
    for w in word:
      if w in vocab:
        vocab[w] = vocab[w]+1
      else:
        vocab[w] = 1

    text_sent = sent_tokenize(text) ## array of sent
    for s in range(len(text_sent)):
      sent = word_tokenize(text_sent[s]) ## simple space tokenizer
      text_sent[s] = ' '.join(sent) ## join back with spacing so word like "wow!" --> "wow !"


    ## keep sent with long enough words ?? because sent_tokenize is not perfect
    bert_pretrain_file.write( "\n".join( text_sent ) ) ## text_sent = [ ["w1 w2 ..."], ["w3 ..."] ]
    bert_pretrain_file.write( "\n\n") ## blank between document

    bert_sentiment_file.write(user_map[row['user_id']] + " " + " ".join(text_sent)+"\n")


  bert_pretrain_file.close()
  bert_sentiment_file.close()


## some summary
print ('total vocab {}'.format(len(vocab)))
vo = sorted ( list(vocab.keys()) )
fout = open('yelp_review_vocab_all.txt',"w")
for v in vo:
  fout.write(v+"\t"+str(vocab[v])+"\n")

fout.close()

