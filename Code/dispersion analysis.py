
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string

from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# df = pd.read_csv(r'avatar corpus.csv')
# df2 = df.dropna(subset=['full_text'])

df = pd.read_csv(r'korra.csv')
df2 = df.dropna(subset=['full_text'])

# df = df.dropna(subset=['text'])



# df= df['text'].tolist()

df2= df2['full_text'].tolist()

def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))



flatten_list=listToString(df2)
# flatten_list = flatten_list.split()




import matplotlib.pyplot as plt
import re

text=flatten_list
words = re.split("\W", text.lower()) # split into words
words = [w for w in words if w != ""] # remove empty elements
WORD = "peace" # define word to search for
# WORD = "killed" # define word to search for

x=list()
for i in range(0,len(words)): # for every word in text
    if words[i] == WORD: # check if word is word we are searching for
        x.append(i) # if so, append its position to variable x

fig, ax = plt.subplots()
ax.vlines(x, 0, 1, edgecolor="red") # <-- ANSWER
ax.set_xlim([0, len(words)]) # set the lower and upper limits of graph
ax.set_xlabel('narrative time')
ax.set_xticks([0],minor=True) # turn off: ax.set_xticks([])
ax.set_ylabel(WORD) # turn off by droping this line
ax.set_yticks([])
fig.set_figheight(1) # figure height, see also fig.set_figwidth()
plt.show()

