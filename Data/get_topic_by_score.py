


import re,sys,os,pickle,gzip
import numpy as np
import pandas as pd


## get list of topic to test on. can only test on reasonable topics


fin = pd.read_csv("/u/scratch/d/datduong/SemEval2017Task4/4B-English/task4B_bert_sentiment_file_mask.txt",sep='\t',dtype=str)

true_label_dict = {} ## 'topic' [negative positive]
for row,line in fin.iterrows():
  if line['tweet_topic'] not in true_label_dict:
    true_label_dict[line['tweet_topic']] = [line['label']] ## collect negative/positive array
  else:
    true_label_dict[line['tweet_topic']].append(line['label']) ## append true_label_dict

#
# count how many positive/negative
true_label_dict_score = {}
for topic in true_label_dict: ## now we actually count
  number_people = len(true_label_dict[topic]) ## for this topic
  negative = true_label_dict[topic].count('not_entailment') ## negative is map to not_entailment, because we want to keep QNLI format
  positive = true_label_dict[topic].count('entailment')
  this_score = np.array ([negative,positive])
  this_score = this_score / np.sum(this_score)
  # major_vote = get_major_vote_topic(true_label_dict[topic])
  true_label_dict_score[topic] = this_score.tolist() + [number_people] # + [major_vote]


## filter

topic_to_test = []
topics = sorted( list ( true_label_dict_score.keys() ) )
for topic in topics:
  if (true_label_dict_score[topic][1] < 0.7) and (true_label_dict_score[topic][1] > 0.3) : ## [1] is fraction of positive
    topic_to_test.append(topic)


##
fout = open("/u/scratch/d/datduong/SemEval2017Task4/4B-English/topic_to_test_3_7.txt","w")
fout.write("\n".join(topic_to_test))
fout.close() 

topic_to_test
