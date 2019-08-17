

## easier to just read in file and mask by column 


## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 

os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
# df = pd.read_csv("task4B_bert_sentiment_nonan_user.txt",sep='\t')

name = 'user_name user_desc user_loc  user_gender'.split()  # tweet_text
name2 = 'name desc loc gender'.split() # text
name_dict = {}
for index,n in enumerate(name): 
  name_dict[n] = name2[index]

for key,val in name_dict.items(): ## @key is "user_name", @val is "name"
  df = pd.read_csv("task4B_bert_sentiment_nonan_user.txt",sep='\t')
  df['tweet_text'] = '[MASK]'
  to_mask = [k for k in name if k!=key] ## mask everything example for @key
  for m in to_mask:
    df[m] = '[MASK]'
  df.to_csv("task4B_bert_sentiment_nonan_user_keep_"+val+"_mask_text.txt",sep="\t",index=None)



### mask all user 
df = pd.read_csv("task4B_bert_sentiment_nonan_user.txt",sep='\t')
name = 'user_name user_desc user_loc  user_gender'.split()
for n in name: 
  df[n] = '[MASK]'


df.to_csv("task4B_bert_sentiment_nonan_user_mask_user_data.txt",sep="\t",index=None)



### COUNT HOW MANY MASK
import os,re,sys,pickle
import pandas as pd 
os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
name = 'user_name user_desc user_loc  user_gender'.split()  # tweet_text
name2 = 'name desc loc gender'.split() # text
name_dict = {}
for index,n in enumerate(name): 
  name_dict[n] = name2[index]

df = pd.read_csv("task4B_bert_sentiment_nonan_user.txt",sep='\t')
for key,val in name_dict.items(): ## @key is "user_name", @val is "name"
  # df = pd.read_csv("task4B_bert_sentiment_nonan_user_keep_"+val+"_mask_text.txt",sep='\t')
  print ('{} count {}'.format(key, list (df[key]).count('[MASK]')))

