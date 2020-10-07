# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:22:02 2020

@author: Ian
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os  # for os.path.basename
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from collections import defaultdict
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

## read csv
## I wanna cluster music of different genres based on 3 musical index
## set labels
df = pd.read_csv('diff_genres.csv')
f = lambda x: x.split(".")[0]
df["genre"] = df['filename'].apply(f)
## label should be numeric
def label_race (row):
   if row['genre'] == 'blues' :
      return 0
   if row['genre'] == 'classical' :
      return 1
   if row['genre'] == 'country' :
      return 2
   if row['genre'] == 'disco':
      return 3
   if row['genre']  == 'hiphop':
      return 4
   if row['genre'] == 'jazz':
      return 5
   if row['genre'] == 'metal':
      return 6
   if row['genre'] == 'pop':
      return 7
   if row['genre'] == 'reggae':
      return 8
   if row['genre'] == 'rock':
      return 9
   return 'Other'
df['label'] = df.apply (lambda row: label_race(row), axis=1)
kmeans = KMeans(n_clusters=10, random_state = 0) ## K-MEAN

##	'mfcc1',	'mfcc2',	'mfcc3',	'mfcc4',	'mfcc5',	'mfcc6'	,'mfcc7',	'mfcc8',	'mfcc9',	'mfcc10',	'mfcc11',	'mfcc12',	'mfcc13',	'mfcc14',	'mfcc15',	'mfcc16',	'mfcc17',	'mfcc18',	'mfcc19',	'mfcc20'
df['cluster'] = kmeans.fit_predict(df[['mfcc1','mfcc2','mfcc3']])
X = df[['mfcc1','mfcc2','mfcc3']]
##X = df[['chroma_stft','spectral_bandwidth','spectral_centroid']]
y = df[['label']]
fignum = 1
## Compare different K
estimators = [KMeans(n_clusters=3),KMeans(n_clusters=5),KMeans(n_clusters=10)]
titles = [ '3 clusters', '5 clusters','10 clusters']
for est in estimators:
  fig = plt.figure(fignum, figsize=(10, 10))
  ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=34)
  est.fit(X)
  labels = est.labels_
## scatter plot 3D
  ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2],
              c=labels.astype(np.float), edgecolor='k')
  ax.w_xaxis.set_ticklabels([])
  ax.w_yaxis.set_ticklabels([])
  ax.w_zaxis.set_ticklabels([])
  ax.set_xlabel('mfcc1')
  ax.set_ylabel('mfcc2')
  ax.set_zlabel('mfcc3')
  ax.set_title(titles[fignum - 1])
##  ax.dist = 12
  fignum = fignum + 1
# Plot the ground truth
fig = plt.figure(fignum, figsize=(6, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=34)

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [0,1,2,3,4,5,6,7,8,9]).astype(np.float)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('mfcc1')
ax.set_ylabel('mfcc2')
ax.set_zlabel('mfcc3')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()

from sklearn.metrics.cluster import normalized_mutual_info_score
print(normalized_mutual_info_score(labels,y.label))



from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
## normalize numeric data
data_scaled = normalize(X)
data_scaled = pd.DataFrame(data_scaled, columns=X.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
## dendrogram plot
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

from sklearn.cluster import AgglomerativeClustering
## 2D dot plot
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)

plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['mfcc1'], data_scaled['mfcc2'], c=cluster.labels_) 



labels = range(1, 50)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
## True position of each dot
plt.scatter(X.iloc[:500,0],X.iloc[:500,1], label='True Position')

## Add text(number) on each dot 
for label, x, y in zip(labels, X.iloc[:50, 0], X.iloc[:50, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

## Plot heatmap
## On the left, you can see the true label strip with different colors
## So you could compare the clustering result and the truth
import seaborn as sns
lut = dict(zip(df.label[400:699].unique(), "ybg"))
rowcolor = df.label.map(lut)
g = sns.clustermap(df.loc[df['genre'].isin(['hiphop','jazz','metal'])].iloc[:,9:12],metric = 'euclidean',standard_scale=1,
                  
                   row_colors = rowcolor)

g_2 = sns.clustermap(df.loc[df['genre'].isin(['hiphop','jazz','metal'])].iloc[:,9:12],metric = 'cosine',standard_scale=1,
                  
                   row_colors = rowcolor)

g_3 = sns.clustermap(df.loc[df['genre'].isin(['hiphop','jazz','metal'])].iloc[:,9:12],metric = 'seuclidean',standard_scale=1,
                
                   row_colors = rowcolor)