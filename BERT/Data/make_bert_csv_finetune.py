



from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/SemEval2017Task4Code/BERT/Data/TextData/'

user_map = {} ## map user {1:someID}
df = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.txt',header=None,sep="\t")
user = list(df[0])
userid = list (df[1])
for i in range(len(user)):
  user_map[user[i]]=userid[i]


vocab = {} ## also count, do this for ALL test/train/dev ?

## do we need BERT for pretrained ? # want: one sentence per line, and one blank line between documents.
## we need BERT for sentiment score

## format is 
## 681563394940473347      amy schumer     negative        @MargaretsBelly Amy Schumer is the stereotypical 1st world Laci Green feminazi. Plus she's unfunny

df = pd.read_csv(main_dir+data_type+'.txt',header=None,sep="\t")
df.columns = ['tweet_id', 'topic', 'sentiment_score', 'tweet_text']

bert_pretrain_file = open(out_dir+data_type+"_bert_pretrain_file.txt","w")

for index,row in tqdm ( df.iterrows() ) :

  text = row['tweet_text'] ##.lower()

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

  ## what if there's only 1 sentence ? split sentence in half 
  if len(text_sent) == 1: 
    temp = deepcopy(text_sent)
    text_sent = text_sent[0]
    temp[0] = text_sent[0:(len(text_sent)//2)]
    temp.append ( text_sent[(len(text_sent)//2)::] ) 
    text_sent = deepcopy ( temp ) 

  ## keep sent with long enough words ?? because sent_tokenize is not perfect
  bert_pretrain_file.write( "\n".join( text_sent ) ) ## text_sent = [ ["w1 w2 ..."], ["w3 ..."] ]
  bert_pretrain_file.write( "\n\n") ## blank between document


bert_pretrain_file.close()


## some summary
print ('total vocab {}'.format(len(vocab)))
vo = sorted ( list(vocab.keys()) )
fout = open('vocab_all.txt',"w")
for v in vo:
  fout.write(v+"\t"+str(vocab[v])+"\n")

fout.close()

