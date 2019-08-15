

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


os.chdir ('/u/scratch/d/datduong/GamergateTweet')

# dict_keys(['retweet_count', 'user', 'created_at', 'place', 'in_reply_to_screen_name', 'truncated', 'favorite_count', 'source', 'id', 'in_reply_to_user_id_str', 'text', 'extended_entities', 'retweeted', 'lang', 'favorited', 'entities', 'retweeted_status', 'contributors', 'in_reply_to_user_id', 'is_quote_status', 'coordinates', 'id_str', 'in_reply_to_status_id_str', 'geo', 'possibly_sensitive', 'in_reply_to_status_id'])

# >>> this_user['user'].keys()
# dict_keys(['description', 'followers_count', 'default_profile_image', 'profile_image_url_https', 'profile_link_color', 'created_at', 'profile_use_background_image', 'time_zone', 'has_extended_profile', 'following', 'translator_type', 'profile_sidebar_border_color', 'follow_request_sent', 'profile_background_tile', 'id', 'utc_offset', 'profile_text_color', 'location', 'lang', 'listed_count', 'friends_count', 'profile_sidebar_fill_color', 'profile_background_color', 'protected', 'profile_image_url', 'profile_banner_url', 'name', 'profile_background_image_url', 'entities', 'notifications', 'url', 'favourites_count', 'statuses_count', 'geo_enabled', 'default_profile', 'is_translator', 'contributors_enabled', 'id_str', 'is_translation_enabled', 'screen_name', 'verified', 'profile_background_image_url_https'])

screen_name_seen = {}
fout = open("user_data.tsv","w",encoding='utf-8')
fout.write("screen_name\tuser_name\tuser_desc\tuser_loc\tuser_gender\n")
user_dict = {}
for line in tqdm ( open('Gamergate.json', 'r') ) :
  ## make a dictionary so we can backtrack the names
  this_user = json.loads(line)
  # this_user['user']['screen_name']
  # this_user['user']['description']
  # this_user['user']['name']
  # this_user['user']['location']
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
    fout.write ( this_user['user']['screen_name'] + "\t" + (str(name or '[MASK]')) + "\t" + (str(desc or '[MASK]')) + "\t" + (str(location or '[MASK]')) + "\t" + gender + "\n" )
  else:
    pass



fout.close() 



