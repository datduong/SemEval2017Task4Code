

## make data for prediction of ANY TOPIC 

# WANT SOMETHING LIKE THIS ::: 
# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  screen_name label

import os,re,sys,pickle
import pandas as pd 

topic_list = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/topic_to_test_3_7.txt",sep="\t",header=None)
topic_list = list (topic_list[0])


df = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/task4B_bert_sentiment_file_mask.txt",sep='\t')
# list(df['user_name']).count('[MASK]')
# list(df['user_desc']).count('[MASK]')
# list(df['user_gender']).count('[MASK]')
# list(df['user_loc']).count('[MASK]')

## must select only data with user name + description at the least ??



## whole data with all user that has some information 
import os,re,sys,pickle
from copy import deepcopy
import pandas as pd 
os.chdir("/u/scratch/d/datduong/SemEval2017Task4/4B-English/")
topic_list = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/ZeroshotExperiment/zeroshot_topic.txt",sep="\t",header=None)
topic_list = list (topic_list[0])
topic_list = topic_list + ['red sox','rolling stones','miss usa','twilight']
print (topic_list)
df = pd.read_csv("task4B_bert_sentiment_nonan_user.txt",sep='\t')
for topic in topic_list : 
  df2 = deepcopy(df)
  df2['tweet_topic']=topic
  df2['tweet_text']='[MASK]' ## new topic so no writing
  df2.to_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/PredictTopicNoNanUser/"+re.sub(" ","_",topic)+".txt",sep="\t",index=None)


