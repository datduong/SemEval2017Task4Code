

## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 

df = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/task4B_bert_sentiment_file_mask.txt",sep='\t')

list(df['user_name']).count('[MASK]')
list(df['user_desc']).count('[MASK]')
list(df['user_gender']).count('[MASK]')
list(df['user_loc']).count('[MASK]')

