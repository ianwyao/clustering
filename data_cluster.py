# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:12:43 2020

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
from wordcloud import WordCloud, STOPWORDS

## In this section, I will cluster reviews of different artists/singers
## And see how each text is related
# =============================================================================
# my_dir_path = "6_texts"
# 
# results = defaultdict(list)
# for file in Path(my_dir_path).iterdir():
#     with open(file, "r") as file_open:
# 
#         results["review"].append(file_open.read())
# df_final = pd.DataFrame(results)
# df_final['artist'] = ['Maroon 5','Madonna','Kesha','Coldplay','Katy Perry','Lady Gaga','Drake','Usher','Michael Jackson','Justin Timberlake','Eminem','Kanye West','The Beatles','Rihanna','Oasis'] 
# 
# ## Load Dataframe Directly 
# =============================================================================

df = pd.read_csv("key_mode_reviews.csv",encoding =  "ISO-8859-1")

# =============================================================================
# select = [3,11,15,17,23,43,58,148,305,357,383,436,420,65,97]
# df = df[df.index.isin(select)]
# =============================================================================

## I just keep ONE artist data if there are several
df = df.drop_duplicates('artist', keep='first')
df_final = df[["artist","review"]]

# df_final_artist.to_csv('final_artist.csv')
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in df_final["review"]:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'review', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
for word in totalvocab_stemmed:
    word = re.sub(r"[^a-zA-Z0-9]+", ' ', word)

for word in totalvocab_tokenized:
    word = re.sub(r"[^a-zA-Z0-9]+", ' ', word)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(df_final["review"]) #fit the vectorizer to "review"

## terms is just a list of the features used in the tf-idf matrix. This is a vocabulary
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
## print(dist)

## K MEAN CLUSTERING !!!!!
from sklearn.cluster import KMeans

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

## generate three clusters
idx = np.argsort(km.cluster_centers_.sum(axis=1))
luc = np.zeros_like(idx)
luc[idx] = np.arange(num_clusters)
clus = km.labels_.tolist()
clusters = luc[clus]
print(clusters)

music = {'artist': df_final['artist'].tolist(), 'review': df_final['review'].tolist(), 'cluster': clusters}
frame = pd.DataFrame(music, index = [clusters] , columns = ['artist', 'review', 'cluster'])

## Apply KMean model:
model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
model.fit(tfidf_matrix)

## Print 20 words that are best related to the "centre" of each cluster
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
 print("Cluster %d:" % i),
 for ind in order_centroids[i, :40]:
   print(" %s" % terms[ind])


MDS()
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print(xs)
print(ys)

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
j_1 = np.arange(20,30)
cluster_names = {0: [terms[order_centroids[0,j]] for j in j_1] ,
                 1: [terms[order_centroids[1,i]] for i in j_1]}


#create data frame that has the result of the MDS plus the cluster numbers and titles
df_1 = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 

#group by cluster
groups = df_1.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(8, 8)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

plt.show() #show the plot
#plt.savefig('clusters_small_noaxes.png', dpi=200)


from scipy.cluster.hierarchy import ward, dendrogram

tfidf_matrix_2 = tfidf_vectorizer.fit_transform(df_final["review"][0:20]) #fit the vectorizer to "review"
dist = 1 - cosine_similarity(tfidf_matrix_2)
linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(5, 5)) # set size
ax = dendrogram(linkage_matrix, orientation="right");

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

## Next I will draw heatmap using 3 different distance metrix
import seaborn as sns
lut = dict(zip(df_1.label.unique(), "yb"))
rowcolor = df_1.label.map(lut)
g = sns.clustermap(df_1.iloc[:,0:2],metric = 'euclidean',standard_scale=1,
                  
                   row_colors = rowcolor)

g = sns.clustermap(df_1.iloc[:,0:2],metric = 'cosine',standard_scale=1,
                  
                   row_colors = rowcolor)

g = sns.clustermap(df_1.iloc[:,0:2],metric = 'seuclidean',standard_scale=1,
                  
                   row_colors = rowcolor)

df['two_labels'] = clusters
itr = 0
comment_words_1 = ''
comment_words_2 = ''
while(itr < 74):
    if (df.iloc[itr,:]['two_labels'] == 0):
        comment_words_1 += df.iloc[itr,:]['review']
    if (df.iloc[itr,:]['two_labels'] == 1):
        comment_words_2 += df.iloc[itr,:]['review']
    itr = itr + 1
# Put all words into comment_words 1 or 2
# Create and generate a word cloud image:

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(comment_words_1)

wordcloud_2 = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words_2) 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
plt.imshow(wordcloud_2, interpolation='bilinear')
plt.axis("off")
plt.show()
