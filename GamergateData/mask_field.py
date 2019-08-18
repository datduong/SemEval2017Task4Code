


import os,re,sys,pickle
import pandas as pd 

os.chdir("/u/scratch/d/datduong/GamergateTweet/SplitData/NotMask/")

df = pd.read_csv("test.tsv",sep='\t')
df['tweet_text'] = '[MASK]'

df.to_csv("test_mask_text.txt",sep="\t",index=None)



