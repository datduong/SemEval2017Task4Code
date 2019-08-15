

import pandas as pd
import os,sys,re,pickle,gzip
import numpy as np
from tqdm import tqdm


server = '/u/scratch/d/datduong/'
# main_dir = '/local/datdb/SemEval2017Task4/4B-English/'

def remove_extreme_topic_add_to_train (topic_list,train,validate,test):
  # @train is panda df.
  ## @topic_list is topic we want to test on. these are topics without extreme vote near 100%

  temp = validate [ validate['tweet_topic'].isin(topic_list) ] ## stuffs to remove and add to @test set
  validate = validate [ ~validate['tweet_topic'].isin(topic_list) ] ## dev has no @topic_list
  test = pd.concat([test, temp]) ## add the topic to be tested from dev into test set. so we have more test sample

  temp = test [ ~test['tweet_topic'].isin(topic_list) ] ## stuffs to remove and add to @train
  test = test [ test['tweet_topic'].isin(topic_list) ] ## retain only topics we want to test on
  train = pd.concat([train, temp])
  return train, validate, test


## note: the ratio are computed to match the baseline ratio
def train_validate_test_split(df, train_percent=.80, validate_percent=.05, seed=1234):
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_percent * m)
  validate_end = int(validate_percent * m) + train_end
  train = df.ix[perm[:train_end]]
  train['index'] = np.arange(0,train.shape[0]) # reset index
  validate = df.ix[perm[train_end:validate_end]]
  validate['index'] = np.arange(0,validate.shape[0])
  test = df.ix[perm[validate_end:]]
  test['index'] = np.arange(0,test.shape[0])
  return train, validate, test



def submitJobs (main_dir, in_file, to_skip, filter_topic, topic_to_test_file, where_save, base_name) :

  if filter_topic == 1:
    print ('\n\nread topic and keep only topic with equal vote in test set\n\n')
    topic_list = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/"+topic_to_test_file+".txt",header=None)
    topic_list = sorted (list (topic_list[0]))
  else:
    topic_list = None


  if to_skip == 'none':
    to_skip = ""
  else:
    to_skip = "_" + re.sub(r"\+","_",to_skip)

  main_dir = server+'SemEval2017Task4/4B-English/'
  os.chdir(main_dir)

  df = pd.read_csv(main_dir+in_file,sep="\t") ## task4B_bert_sentiment_file_full.txt
  main_dir = main_dir + '/' + where_save
  if not os.path.exists(main_dir):
    os.mkdir(main_dir)


  for fold in [1]:
    where_fold = os.path.join( main_dir , base_name + to_skip )
    if not os.path.exists(where_fold):
      os.mkdir(where_fold)
    os.chdir(where_fold)

    train, validate, test = train_validate_test_split(df,seed=int(1234/fold))

    ## fix train/validate/test because some topic has too many positive vote
    if topic_list is not None:
      train,validate,test = remove_extreme_topic_add_to_train (topic_list,train,validate,test)

    train.to_csv('train.tsv',index=None,sep="\t")
    validate.to_csv('dev.tsv',index=None,sep="\t")
    test.to_csv('test.tsv',index=None,sep="\t")



if len(sys.argv)<1: ## run script
  print("Usage: \n")
  sys.exit(1)
else:
  submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7] )

