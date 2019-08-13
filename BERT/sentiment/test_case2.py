

import pandas as pd 

df = pd.read_csv("task4B_bert_sentiment_file_mask.txt",sep="\t")

s = list (df ['tweet_id']) 
g = []
for x in s: 
  if x in g: 
    print (x)
    break 
  else: 
    g.append(x)

## !! same tweet id ??

625548653596282880      ant-man positive        "Going to the movies tomorrow to watch Southpaw, Paper Towns, Train Wreck and then Ant-Man again."
625548653596282880      paper towns     positive        "Going to the movies tomorrow to watch Southpaw, Paper Towns, Train Wreck and then Ant-Man again."

