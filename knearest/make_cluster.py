
import os,sys,re,pickle,gzip

from collections import Counter

import numpy as np
import pandas as pd

from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from scipy import spatial

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

colors = mcolors.TABLEAU_COLORS
list(colors.items())

os.chdir('/u/scratch/d/datduong/SemEval2017Task4/4B-English/BertSentiment')

def closest_node(node, nodes, nodes_names):
  ## node is "centroid" or some center point for cluster
  ## use [node] to make 1D array into 2D
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
  dist = spatial.distance.cdist([node], nodes, metric='euclidean')
  where = np.argsort(dist.flatten())  ## smallest to largest (15 items)
  return nodes_names[where[0:3]] , dist.flatten()[where[0:3]]


def text_2_np_user (input_name):
  user_vector = []
  user_name = []
  fin = open(input_name,"r")
  for line in fin:
    user = line.strip().split(" ")
    if re.match( r'^userId[0-9]+', user[0]): ## get user only
      user_name.append(user[0])
      vector = [float(x) for x in user[1::]]
      user_vector.append(vector)
  fin.close()
  return np.array (user_vector), user_name


def text_2_np_concept (input_name,concept_words):
  user_vector = []
  user_name = []
  fin = open(input_name,"r")
  for line in fin:
    user = line.strip().split(" ")
    if user[0] in concept_words: ## get only words we want
      user_name.append(user[0])
      vector = [float(x) for x in user[1::]]
      user_vector.append(vector)
  fin.close()
  return np.array (user_vector), user_name

def extract_user_in_group (user_name, user_cluster, group_num):
  ## !! extract user ??
  where = np.where ( user_cluster == group_num )[0]
  return np.array(user_name)[where]



## normalize user_vector to norm 1 ? well, this depends on the type of metrics

## **** KMEAN CLUSTERING

user_vector , user_name = text_2_np_user( "word_vector.txt" )

n_clusters=10

colors=list(mcolors.TABLEAU_COLORS.items())
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(user_vector)
clusters = kmeans.predict(user_vector)
# kmeans.cluster_centers_ ## get centers position

print(len(clusters))
print(Counter(clusters))
num_per_cluster = Counter(clusters)
for c in range(10):
  print (num_per_cluster[c])


## get concept vectors closest to the centroids

word_list = ['misogynistic','feminism','trump','texas','california','democratic','republican']

concept_vector, concept_name = text_2_np_concept( "word_vector.txt", word_list )

closest_word = []
distance = []
for i in range(n_clusters):
  print ('\n\ncluster {}'.format(i))
  out, dist = closest_node(kmeans.cluster_centers_[i], concept_vector, np.array(concept_name))
  closest_word.append(out)
  distance.append(dist)

closest_word = np.array (closest_word)
np.savetxt('closestWordsKmean.txt',closest_word.T,fmt='%s')

distance = np.array(distance)
np.savetxt('closestWordsKmeanDistance.txt',distance.T,fmt='%s')

## PCA, normalize each column to mean 0 variance 1
# user_vector_standardize = (user_vector - user_vector.mean(axis=0)) / user_vector.std(axis=0)


## extract user group

fin = pd.read_csv('/u/scratch/d/datduong/SemEval2017Task4/4B-English/output_semeval_tweet_userinfo.gender.tsv',sep="\t",dtype='str')
user_counter = {}
for index,row in fin.iterrows():
  if row['user_id'] in user_counter: 
    user_counter[row['user_id']] = user_counter[row['user_id']] + 1
  else: 
    user_counter[row['user_id']] = 1

## count tweet per user 
user_num = np.array([user_counter[k] for k in user_counter])
print (np.quantile(user_num, np.arange(0.1,1,.1)))
from scipy import stats
stats.describe(user_num)

for group_num in [6,7]:
  user_group_to_get = extract_user_in_group(user_name, clusters,group_num)
  user_group_to_get = [ re.sub("userId","",u) for u in user_group_to_get] ## get row number
  fin2 = fin[fin['user_id'].isin(user_group_to_get)]
  fin2.to_csv("group_to_check"+str(group_num)+".tsv",index=None,sep="\t")

