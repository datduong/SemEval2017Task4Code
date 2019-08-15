
import os,sys,re,pickle
import numpy as np
import pandas as pd
## extract only user in test set, see what is accuracy 

main_dir = '/u/scratch/d/datduong/SemEval2017Task4/4B-English'
os.chdir(main_dir)

user_df = pd.read_csv("output_semeval_tweet_userinfo.gender.tsv",sep="\t") ## has both user+tweet 
tweet_with_user = list ( user_df['tweet_id'] )

# 'full_data_mask_name_description_location_user_gender','full_data_mask','full_data_mask_text','full_data_mask_description' 'full_data_mask_description_location_user_gender','full_data_mask_name_description_location_user_gender','full_data_mask_user_gender','full_data_mask_text','full_data_mask_name','full_data_mask_location','full_data_mask_description','full_data_mask_description_name','full_data_mask_description_user_gender','full_data_mask_description_location'
for folder in ['full_data_mask_location_user_gender','add_gamergate']: 
  test_df = pd.read_csv('BertSentimentFilterTestLabel37/'+folder+"/test.tsv",sep="\t")
  print ('num row 1st read in {} '.format(test_df.shape))
  test_df = test_df [ test_df['tweet_id'].isin(tweet_with_user)]
  print ('num row keep only user {} '.format(test_df.shape))
  test_df.to_csv('BertSentimentFilterTestLabel37/'+folder+"/test_user_only.tsv",sep="\t",index=None)


# ## filter by topic, some topic has almost 100% positive. 

# for topic in ['islam','israel','sharknado']: 
#   for folder in ['full_data_mask_name_description_location_user_gender','full_data_mask','full_data_mask_text','full_data_mask_name','full_data_mask_description','full_data_mask_location','full_data_mask_user_gender']: 
#     test_df = pd.read_csv('BertSentimentFilterTestLabel/'+folder+"/test.tsv",sep="\t")
#     print ('num row 1st read in {} '.format(test_df.shape))
#     test_df = test_df [ test_df['tweet_id'].isin(tweet_with_user)]
#     test_df = test_df [ test_df['tweet_topic'].isin([topic])] 
#     print ('num row keep only user {} '.format(test_df.shape))
#     test_df.to_csv('BertSentimentFilterTestLabel/'+folder+"/test_user_only_"+topic+".tsv",sep="\t",index=None)



# ## we will extract topics where fraction positive/negative are borderline. 

