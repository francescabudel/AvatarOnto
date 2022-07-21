
from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import re
import string

# from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# df = pd.read_csv(r'avatar corpus.csv')

df = pd.read_csv(r'korra.csv')

# df2 = df.dropna(subset=['full_text'])


# df = df.loc[df['character'] == 'Ozai']
# df = df.loc[df['book'] != 'Fire']
df = df.dropna(subset=['text'])

def percentage(part,whole):
 return 100 * float(part)/float(whole)


count = 0
polarity = 0



#Creating new dataframe and new features
tw_list = pd.DataFrame(df)

remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()



tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp



tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]





def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])



print(count_values_in_column(tw_list,"sentiment"))

pichart = count_values_in_column(tw_list, "sentiment")
names = pichart.index

size = pichart["Percentage"]


pos = size['positive'].tolist()
nue = size['neutral'].tolist()
neg = size['negative'].tolist()



labels = ['Neutral ['+str(nue)+'%]','Positive ['+str(pos)+'%]','Negative ['+str(neg)+'%]']
plt.legend(labels)

my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(size, labels=labels, colors=[ 'blue','green', 'red'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()




def create_wordcloud(text):
    mask = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                  mask = mask,
                  max_words=3000,
                  stopwords=stopwords,
                  repeat=True)
    wc.generate(str(text))
    wc.to_file("wc.png")
    print("Word Cloud Saved Successfully")
    path="wc.png"
    display(Image.open(path))




create_wordcloud(tw_list["text"].values)


create_wordcloud(tw_list_positive["text"].values)


create_wordcloud(tw_list_negative["text"].values)



create_wordcloud(tw_list_neutral["text"].values)


tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)


round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2)


#Removing Punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))



def tokenization(text):
    text = re.split('\W+', text)
    return text

tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))


stopword = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))




def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # 
    return text

print(tw_list.head())

#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
print('{} Number of sentences has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
print(count_vect_df.head())



# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
print(countdf[1:11])
words_found=[]
count_of_words=[]

for i in countdf[1:8].values.tolist():
    count_of_words.append(int(i[0]))


for i in list(countdf[1:8].index):
    words_found.append(str(i))

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10, 5))

plt.bar(words_found,count_of_words ,color ='maroon',
        width = 0.4)
plt.xlabel("Words")
plt.ylabel("Count of words")
plt.show()

#Function to ngram
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]



#n2_bigram
n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)


print(n2_bigrams)
words_found=[]
count_of_words=[]

for i in n2_bigrams[0:7]:
    words_found.append(i[0])
    count_of_words.append(i[1])




fig = plt.figure(figsize = (14, 5))

plt.bar(words_found,count_of_words ,color ='purple',
        width = 0.4)
plt.xlabel("Bi-grams")
plt.ylabel("Count of occurances")
plt.show()


#n3_trigram
n3_trigrams = get_top_n_gram(tw_list['text'],(3,3),20)

words_found = []
count_of_words = []

for i in n3_trigrams[0:7]:
    words_found.append(i[0])
    count_of_words.append(i[1])

fig = plt.figure(figsize=(25, 5))

plt.bar(words_found, count_of_words, color='purple',
        width=0.4)
plt.xlabel("Ti-grams")
plt.ylabel("Count of occurances")
plt.show()

print(n3_trigrams)




