

import os,sys,re,pickle
import pandas as pd 
import numpy as np 


os.chdir('/u/scratch/d/datduong/GamergateTweet')

user_data = pd.read_csv("GamergateTweetTextUserData.tsv",sep='\t') ## stupid quotes
with open("GamergateTweetTextUserData.tsv", "r", encoding="utf-8-sig") as f:
  reader = csv.reader(f, delimiter="\t", quotechar=None,quoting=csv.QUOTE_NONE)
  lines = []
  for line in reader:
    if sys.version_info[0] == 2:
      line = list(unicode(cell, 'utf-8') for cell in line)
    lines.append(line)


user_data = pd.read_csv("user_data.tsv",sep='\t',quoting=3) ## stupid quotes
# screen_name name  description location  gender

feminist_usernames = pd.read_csv("feminist_usernames",sep="\t",header=None)
feminist_usernames.columns=['screen_name']
feminist_usernames['user_group'] = 'feminist'
feminist_usernames['label'] = 'not_entailment' # not_entailment' # 'negative'


misogynist_usernames = pd.read_csv("misogynist_usernames",sep="\t",header=None)
misogynist_usernames.columns=['screen_name']
misogynist_usernames['user_group'] = 'misogynist'
misogynist_usernames['label'] = 'entailment' # 'entailment' # 'positive'


usernames = pd.concat([feminist_usernames,misogynist_usernames])

Mturk_feminist_comments_format = pd.read_csv("Mturk_feminist_comments_format_short.txt",sep="\t",header=None) ## feminist users oppose to gamergate
Mturk_misogynist_comments_format = pd.read_csv("Mturk_misogynist_comments_format_short.txt",sep="\t",header=None) ## misogynists support gamergate

comments = pd.concat([Mturk_feminist_comments_format,Mturk_misogynist_comments_format])
comments.columns=['tweet_text']
usernames_comment = pd.concat([usernames,comments],axis=1)

user_data = pd.merge(user_data,usernames_comment,on='screen_name')
user_data['tweet_topic'] = 'gamergate'
user_data['index'] = np.arange(0,user_data.shape[0])
user_data['tweet_id'] = np.arange(0,user_data.shape[0])

# index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id  label

name = 'index user_name user_desc user_loc  user_gender tweet_topic tweet_text  tweet_id screen_name  label'.split() 
user_data = user_data[name]

user_data.to_csv("GamergateTweetTextUserData.tsv",sep="\t",index=None)

## some user are not in user_data. so we remove their tweets. example screen name "iRickDaKid" is only appearing in the "in-reply-to"
not_in = [s for s in list(usernames_comment['screen_name']) if s not in list(user_data['screen_name'])]

