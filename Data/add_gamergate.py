

## easier to just read in file and mask by column 


## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 
import numpy as np

# os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
# df = pd.read_csv("task4B_bert_sentiment_file_mask.txt",sep='\t')
# gamergate = pd.read_csv("/u/scratch/d/datduong/GamergateTweet/GamergateTweetTextUserData.tsv",sep="\t")
# names = 'index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id label'.split()
# gamergate = gamergate[names]
# ## add 1000 
# gamergate = gamergate.sample(n=1000)

# df = pd.concat([df,gamergate])
# df.to_csv("task4B_bert_sentiment_add_gamergate.txt",sep="\t",index=None)


### !! add proper train file of gamergate 

os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
df = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUserZeroshot/mask_text/train.tsv",sep='\t')

gamergate = pd.read_csv("/u/scratch/d/datduong/GamergateTweet/SplitData/NotMask/train.tsv",sep="\t")
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label
gamergate = gamergate.dropna() # remove anything that is Nan , why is there still Nan? there already MASK

names = 'index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id label'.split()
gamergate = gamergate[names]

## mask text 
gamergate['tweet_text'] = '[MASK]'

df = pd.concat([df,gamergate])
df['index'] = np.arange(0,df.shape[0]) ## reorder row 1 2 3 4 ...
df.to_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUserZeroshot/mask_text/train_add_gamergate.txt",sep="\t",index=None)


