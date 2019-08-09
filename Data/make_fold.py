

import pandas as pd 
import os,sys,re,pickle,gzip 
import numpy as np 
from tqdm import tqdm 

main_dir = '/local/datdb/SemEval2017Task4/4B-English/'
os.chdir(main_dir)


## note: the ratio are computed to match the baseline ratio
def train_validate_test_split(df, train_percent=.85, validate_percent=.05, seed=1234):
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



df = pd.read_csv("/local/datdb/SemEval2017Task4/4B-English/task4B_bert_sentiment_file_notweet.txt",sep="\t")

main_dir = main_dir + '/' + 'BertSentiment' 

if not os.path.exists(main_dir): 
  os.mkdir(main_dir)


for fold in [1]: 
  where_fold = os.path.join( main_dir , "notweet_fold_"+str(fold))
  if not os.path.exists(where_fold): 
    os.mkdir(where_fold)
  os.chdir(where_fold)
  train, validate, test = train_validate_test_split(df,seed=int(1234/fold))
  train.to_csv('train.tsv',index=None,sep="\t")
  validate.to_csv('dev.tsv',index=None,sep="\t")
  test.to_csv('test.tsv',index=None,sep="\t")
