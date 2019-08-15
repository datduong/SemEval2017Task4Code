
import os,sys,re,pickle
from collections import OrderedDict

## 

os.chdir('/u/scratch/d/datduong/GamergateTweet')

def format_tweet_data(fin_name,fout_name): 
  fout = open(fout_name,"w") # "Mturk_feminist_comments_format.txt"
  fin = open(fin_name,'r') # "Mturk_feminist_comments"
  for counter, line in enumerate(fin):  
    line = line.replace("|********************","").replace("********************|","").strip() 
    line = re.sub(r"^rt","",line).strip() 
    line = line.split(" rt ") ## split by retweet
    line = " . ".join(s for s in list(OrderedDict.fromkeys(line))) ## remove duplicated tweets because of retweet
    line = re.sub(r'https?://\S+', " " ,line).strip() 
    # if counter > 1: 
    #   break
    line = line.split() ## some tweets are too long ??? 
    if len(line) > 150: 
      line = line[0:150]
    line = " ".join(line)
    fout.write(line+"\n")
  #
  fout.close() 


format_tweet_data("Mturk_feminist_comments","Mturk_feminist_comments_format_short.txt")

format_tweet_data("Mturk_misogynist_comments","Mturk_misogynist_comments_format_short.txt")

