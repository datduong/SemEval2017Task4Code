
import os,sys,re,pickle
import numpy as np
import pandas as pd
## extract only user in test set, see what is accuracy 

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English'
os.chdir(main_dir)

user_df = pd.read_csv("output_semeval_tweet_userinfo.gender.tsv",sep="\t") ## has both user+tweet 
tweet_with_user = list ( user_df['tweet_id'] )

for folder in ['full_data_mask_type_name_description_location_user_gender','full_data_mask_type','full_data_mask_type_text','full_data_mask_type_name','full_data_mask_type_description','full_data_mask_type_location','full_data_mask_type_user_gender']: 
  test_df = pd.read_csv('BertSentiment/'+folder+"/test.tsv",sep="\t")
  print ('num row 1st read in {} '.format(test_df.shape))
  test_df = test_df [ test_df['tweet_id'].isin(tweet_with_user)]
  print ('num row keep only user {} '.format(test_df.shape))
  test_df.to_csv('BertSentiment/'+folder+"/test_user_only.tsv",sep="\t",index=None)

