



import json,os,sys,re,gzip,pickle
from tqdm import tqdm
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
import gender_guesser.detector as gender

GenderDetector = gender.Detector(case_sensitive=False)

def GetGender (name):
  # name = name.split()
  name = word_tokenize(name)
  if len(name) == 1:
    gender = GenderDetector.get_gender(name[0]) ## only 1 entry
  else:  # long or weird names
    gender = " ".join([ GenderDetector.get_gender(n) for n in name ]) ## we don't know how to average this ??
  ##
  gender = re.sub ("female","FEMALE",gender) ## will make regx easier later
  gender = re.sub('unknown',' ',gender).strip()
  if len(gender) == 0:
    gender='[MASK]'
  else:
    gender = gender.split()[0].lower().replace("_", " ") ## first occurrance. for example, Mary Jonhson, will be "female male", but we want only the female
  return gender


os.chdir ('/local/datdb/TweetShootData2018')

list_name = """nashville
pittsburgh
santa_fe
thousand_oaks
dallas
colorado_springs
chattanooga
burlington
baton_rouge
fresno
fort_lauderdale
roseburg
parkland
orlando
kalamazoo
sutherland_springs
san_francisco
san_bernardino
vegas
thornton
annapolis
"""

list_name = list_name.split() 


for add_on in list_name :

  print ('name {}'.format(add_on))

  json_path = '/local/jyzhao/Github/data/tweets/'+add_on+'/'+add_on+'.json'

  screen_name_seen = {}

  fout = open("user_data_with_tweet_"+add_on+".tsv","w",encoding='utf-8')
  fout.write("screen_name\tuser_name\tuser_desc\tuser_loc\tuser_gender\ttweet_text\n")
  # user_dict = {}
  for line in tqdm ( open(json_path, 'r') ) :
    ## make a dictionary so we can backtrack the names
    this_user = json.loads(line)
    # break

    if this_user['retweeted'] : 
      continue ## don't need to count retweets ?? 

    # this_user['user']['screen_name']
    # this_user['user']['description']
    # this_user['user']['name']
    # this_user['user']['location']

    ## tweet text ??
    tweet_text = this_user['full_text'] ## got truncated ??
    if 'retweeted_status' in this_user:
      tweet_text = this_user['retweeted_status']['full_text']

    tweet_text = tweet_text.replace("\n"," ").replace("\t"," ").replace("\r"," ").replace('"'," ").strip() ## clean tweet text 

    if this_user['user']['screen_name'] not in screen_name_seen:
      screen_name_seen[this_user['user']['screen_name']] = 1 ## some user appear several times ???
      gender = GetGender(this_user['user']['name'])
      # fout.write("\t".join( this_user['user'][key].replace("\n"," ").replace("\t"," ").replace("\r"," ") for key in ['screen_name','name','description','location'] ) + "\t" + gender + "\n" )
      # break
      name = this_user['user']['name'].replace("\n"," ").replace("\t"," ").replace("\r"," ").replace('"'," ").strip()
      desc = this_user['user']['description'].replace("\n"," ").replace("\t"," ").replace("\r"," ").replace('"'," ").strip()
      location = this_user['user']['location'].replace("\n"," ").replace("\t"," ").replace("\r"," ").replace('"'," ").strip()
      # if this_user['user']['screen_name'] == 'Mister_havoc':
      #   break
      fout.write ( this_user['user']['screen_name'] + "\t" + (str(name or '[MASK]')) + "\t" + (str(desc or '[MASK]')) + "\t" + (str(location or '[MASK]')) + "\t" + gender + "\t" + tweet_text + "\n" )
    else:
      pass

  fout.close()





