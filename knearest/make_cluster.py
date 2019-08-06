
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
    if re.match( r'^user[0-9]+', user[0]): ## get user only
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

