

import re,sys,os,pickle,gzip
import numpy as np
import pandas as pd

## for each topic, count fraction positive/negative

def simple_accuracy(preds, labels):
  return (preds == labels).mean()

def get_major_vote_topic(label_list):
  num_negative = label_list.count('not_entailment') ## negative is map to not_entailment, because we want to keep QNLI format
  num_positive = label_list.count('entailment')
  major_vote = 'not_entailment'
  if num_negative < num_positive:
    major_vote = 'entailment'
  return major_vote

def accuracy_by_topic (true_label_arr_train,true_label_arr_test): # @true_label is array
  major_vote = get_major_vote_topic(true_label_arr_train)
  # prediction takes on @major_vote
  prediction = [major_vote] * len(true_label_arr_test)
  return simple_accuracy (np.array(prediction), np.array(true_label_arr_test))


folder = 'MaskText'
fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUser/"+folder+"/train.tsv",sep='\t',dtype=str)
# fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/task4B_bert_sentiment_file_mask.txt",sep='\t',dtype=str)

true_label_dict_train = {} ## 'topic' [negative positive]
for row,line in fin.iterrows():
  if line['tweet_topic'] not in true_label_dict_train:
    true_label_dict_train[line['tweet_topic']] = [line['label']] ## collect negative/positive array
  else:
    true_label_dict_train[line['tweet_topic']].append(line['label']) ## append true_label_dict_train

#
# count how many positive/negative
output = {}
for topic in true_label_dict_train: ## now we actually count
  number_people = len(true_label_dict_train[topic]) ## for this topic
  negative = true_label_dict_train[topic].count('not_entailment') ## negative is map to not_entailment, because we want to keep QNLI format
  positive = true_label_dict_train[topic].count('entailment')
  this_score = np.array ([negative,positive])
  this_score = this_score / np.sum(this_score)
  major_vote = get_major_vote_topic(true_label_dict_train[topic])
  output[topic] = this_score.tolist() + [number_people] + [major_vote]


fout = open("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUser/"+folder+'/majority_baseline_by_topic.tsv','w')
# fout = open("/u/scratch/d/datduong/SemEval2017Task4/4B-English/majority_baseline_by_topic_whole_data.tsv",'w')
for key,val in output.items():
  fout.write(key+"\t"+str(val[0])+"\t"+str(val[1])+"\t"+str(val[2])+"\t"+str(val[3])+"\n")

fout.close()


## compute results for test set based on majority vote

fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUser/"+folder+"/test.tsv",sep='\t',dtype=str)

true_score = []
predict_score = []
true_label_dict_test = {}
for row,line in fin.iterrows():
  predict_score.append ( np.argmax ( output [ line['tweet_topic'] ][0:2] ) ) ## just pick best true_label_dict_train from train data
  if line['label'] == 'entailment':
    true_score.append (1)
  else:
    true_score.append (0)
  if line['tweet_topic'] not in true_label_dict_test:
    true_label_dict_test[line['tweet_topic']] = [line['label']] ## collect negative/positive array
  else:
    true_label_dict_test[line['tweet_topic']].append(line['label']) ## append true_label_dict_train


print ('acc on whole data {}'.format(simple_accuracy(np.array(predict_score), np.array(true_score))) )

topic_to_test = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/topic_to_test_3_7.txt",sep="\t",header=None)
topic_to_test = list (topic_to_test[0])

output_test = {}
for topic in true_label_dict_test:
  if topic not in topic_to_test: 
    continue
  ## add count
  number_people = len(true_label_dict_test[topic]) ## for this topic
  negative = true_label_dict_test[topic].count('not_entailment') ## negative is map to not_entailment, because we want to keep QNLI format
  positive = true_label_dict_test[topic].count('entailment')
  this_score = np.array ([negative,positive])
  this_score = this_score / np.sum(this_score)
  major_vote = get_major_vote_topic(true_label_dict_test[topic])
  acc = accuracy_by_topic (true_label_dict_train[topic],true_label_dict_test[topic])
  output_test[topic] = this_score.tolist() + [number_people] + [major_vote] + [acc]



fout = open("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUser/"+folder+'/majority_baseline_by_topic_testset.tsv','w')
for key,val in output_test.items():
  fout.write(re.sub(" ","_",key) +"\t"+ "\t".join(str(s) for s in val) +"\n")

#
fout.close()


# fout = open("/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentimentNoNanUser/"+folder+'/topic_to_test.tsv','w')
# for key,val in output_test.items():
#   fout.write(key+"\n")

# #
# fout.close()

