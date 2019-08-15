

## easier to just read in file and mask by column 


## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 

os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
df = pd.read_csv("task4B_bert_sentiment_file_mask.txt",sep='\t')

df['user_gender'] = '[MASK]'
df['user_loc'] = '[MASK]'

df.to_csv("task4B_bert_sentiment_file_mask_location_user_gender.txt",sep="\t",index=None)


