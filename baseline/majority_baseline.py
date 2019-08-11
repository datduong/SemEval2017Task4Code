

import re,sys,os,pickle,gzip
import numpy as np
import pandas as pd

## for each topic, count fraction positive/negative

folder = 'full_data_mask_type'
fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentiment/"+folder+"/train.tsv",sep='\t',dtype=str)

score = {} ## 'topoic' [negative positive]
for row,line in fin.iterrows():
  if line['tweet_topic'] not in score:
    score[line['tweet_topic']] = [line['label']] ## collect negative/positive array
  else:
    score[line['tweet_topic']].append(line['label']) ## append score

#

# count how many positive/negative
for topic in score: ## now we actually count
  negative = score[topic].count('not_entailment') ## negative is map to not_entailment, because we want to keep QNLI format
  positive = score[topic].count('entailment')
  this_score = np.array ([negative,positive])
  this_score = this_score / np.sum(this_score)
  score[topic] = this_score


## 
# pickle.dump(score,open("majority_score_by_topic.pickle","wb"))

## ****

## compute score based on majority vote

def simple_accuracy(preds, labels):
  return (preds == labels).mean()


def acc_and_f1(preds, labels):
  acc = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds)
  return {
    "acc": acc,
    "f1": f1,
    "acc_and_f1": (acc + f1) / 2,
  }



fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentiment/"+folder+"/test_user_only.tsv",sep='\t',dtype=str)

true_score = []
predict_score = []
for row,line in fin.iterrows():
  predict_score.append ( np.argmax ( score [ line['tweet_topic'] ] ) ) ## just pick best score
  if line['label'] == 'entailment':
    true_score.append (1)
  else: 
    true_score.append (0)


simple_accuracy(np.array(predict_score), np.array(true_score))

fout = open('majority_baseline_by_topic.tsv','w')
for key,val in score.items(): 
  fout.write(key+"\t"+str(val[0])+"\t"+str(val[1])+"\n")

fout.close() 

