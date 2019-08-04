



from __future__ import absolute_import, division, print_function

from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
import argparse,csv,logging,os,random,sys,pickle,gzip,json,re
import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'
out_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English/'


## do we need BERT for pretrained ? # want: one sentence per line, and one blank line between documents.
## we need BERT for sentiment score

## format is 
## 681563394940473347      amy schumer     negative        @MargaretsBelly Amy Schumer is the stereotypical 1st world Laci Green feminazi. Plus she's unfunny

df = pd.read_csv(main_dir+'SemEval2017-task4-dev.subtask-BD.english.INPUT.tsv',header=None,sep="\t")
df.columns = ['tweet_id', 'topic', 'sentiment_score', 'tweet_text']

bert_pretrain_file = open(out_dir+"task4B_bert_pretrain_file.txt","w")

for index,row in tqdm ( df.iterrows() ) :

  text = row['tweet_text'] ##.lower()
  text_sent = sent_tokenize(text) ## array of sent

  ## what if there's only 1 sentence ? split sentence in half 
  if len(text_sent) == 1: 
    temp = deepcopy(text_sent)
    text_sent = text_sent[0].split() ## split space to get whole word
    temp[0] = " ".join ( text_sent[0:(len(text_sent)//2)] ) 
    temp.append ( " ".join ( text_sent[(len(text_sent)//2)::] ) )
    text_sent = deepcopy ( temp ) 

  ## keep sent with long enough words ?? because sent_tokenize is not perfect
  bert_pretrain_file.write( "\n".join( s.strip() for s in text_sent ) ) ## text_sent = [ ["w1 w2 ..."], ["w3 ..."] ]
  bert_pretrain_file.write( "\n\n") ## blank between document


bert_pretrain_file.close()

