

## easier to just read in file and mask by column 


## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 

os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
df = pd.read_csv("task4B_bert_sentiment_file_mask.txt",sep='\t')

gamergate = pd.read_csv("/u/scratch/d/datduong/GamergateTweet/GamergateTweetTextUserData.tsv",sep="\t")
names = 'index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id label'.split()
gamergate = gamergate[names]
## add 1000 
gamergate = gamergate.sample(n=1000)

df = pd.concat([df,gamergate])
df.to_csv("task4B_bert_sentiment_add_gamergate.txt",sep="\t")

