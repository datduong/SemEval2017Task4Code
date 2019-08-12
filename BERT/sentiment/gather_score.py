import sys,re,os,pickle
import numpy as np
import pandas as pd

## we test many topis individually, we have to read through their log files.

main_dir = '/local/datdb/SemEval2017Task4/4B-English/BertSentiment/'

folder_type = 'full_data_mask'+"/by_topic"
base_name = 'test_full_data_mask_name_description_location_user_gender_test_user_only_'

os.chdir(main_dir+folder_type)
file_list = os.listdir(main_dir+folder_type)

topic_list = pd.read_csv("/local/datdb/SemEval2017Task4/4B-English/BertSentiment/full_data_mask/topic_to_test.tsv",sep="\t",header=None)
topic_list = list (topic_list[0])
topic_list_name = [re.sub(" ","_",top) for top in topic_list]


acc_by_topic = {}
for fname in file_list:
  for index, topic in enumerate(topic_list_name) :
    if base_name+topic in fname:
      fin = open(fname,'r')
      for line in fin:
        if "{'acc_':" in line: ## this is the key line we want, because we only care about acc. 
          line = re.sub(",","",line)
          line = line.strip().split()
          acc_by_topic[topic_list[index]] = line[1] ## just write out the score
      fin.close() 

fout = open(base_name+"by_topic.txt",'w')
topic_list.sort()
for top in topic_list: 
  fout.write ( top + "\t" + acc_by_topic[top] + "\n") 

#
fout.close()

      
