

import os,sys,re,pickle
import pandas as pd 
import numpy as np

import scipy
import scipy.stats

def get_spearman_cor (fin,topic_list): 
  prob = fin [ topic_list ].to_numpy()
  return np.round(scipy.stats.spearmanr(prob)[0],4)


topic_list = pd.read_csv("/local/datdb/SemEval2017Task4/4B-English/ZeroshotExperiment/zeroshot_topic.txt",sep="\t",header=None)
topic_list = list (topic_list[0])
topic_list = topic_list + ['red sox','rolling stones','miss usa','twilight'] # red sox','rolling stones','miss usa','twilight
print (topic_list)

topic_list = [ re.sub(" ","_",top) for top in topic_list ]

os.chdir("/local/datdb/SemEval2017Task4/4B-English/BertSentimentNoNanUser/mask_text/PredictTopic")

df = pd.read_csv(topic_list[0]+'.tsv', sep="\t", header=None)
df.columns = [topic_list[0]]

for top in topic_list[1::] : ## get the rest
  df2 = pd.read_csv(top+'.tsv', sep="\t", header=None)
  df2.columns = [top]
  df = pd.concat([df,df2],axis=1)



main_df = pd.read_csv('/local/datdb/SemEval2017Task4/4B-English/task4B_bert_sentiment_nonan_user.txt', sep="\t")
main_df = pd.concat([main_df,df],axis=1)

## whole data
# prob = df.to_numpy()
# spearman_cor = scipy.stats.spearmanr(prob)

# is_female = [ bool(re.match('female',g)) for g in main_df['user_gender'] ]
# is_male = [ bool(re.match('(^male| male )',g)) for g in main_df['user_gender'] ]

# female = main_df[is_female]
# male = main_df[is_male]

# female_cor = get_spearman_cor(female,topic_list)
# male_cor = get_spearman_cor(male,topic_list)
# np.savetxt('female_cor.txt', female_cor) 
# np.savetxt('male_cor.txt', male_cor) 


def split_by_sentiment (main_df,topic,topic_list): 
  positive = main_df[ main_df['tweet_topic'].isin([topic]) & (main_df['label']=='entailment') ]
  ## we know what the true label is if we split by group 
  # positive[topic]
  # we can't have all 0's or all 1's because we will get NaN in rank-correlation
  # positive[topic] = 1.0 ## high prob is "positive"
  negative = main_df[ main_df['tweet_topic'].isin([topic]) & (main_df['label']=='not_entailment') ]
  # negative[topic] = 0.0
  print ('dim pos df {}'.format(positive.shape))
  print ('dim neg df {}'.format(negative.shape))
  positive = get_spearman_cor(positive,topic_list)
  negative = get_spearman_cor(negative,topic_list)
  return positive, negative



pos_out,neg_out = split_by_sentiment(main_df,'christians',topic_list)
np.savetxt('christians_pos_cor.txt', pos_out) 
np.savetxt('christians_neg_cor.txt', neg_out) 


