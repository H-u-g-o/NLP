import pandas as pd
import gensim 
import logging
import  re
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict 
from gensim.models import Word2Vec
import multiprocessing
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tweet = pd.read_csv('tweet.csv')

stop_words = stopwords.words("english")

def string_manipulation(df,column)  :
    #extract hashtags
    df["hashtag"]  = df[column].str.findall(r'#.*?(?=\s|$)')
    #extract twitter account references
    df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')

    #remove hashtags and accounts from tweets
    df[column] = df[column].str.replace(r'@.*?(?=\s|$)'," ")
    df[column] = df[column].str.replace(r'#.*?(?=\s|$)'," ")

    #convert to lower case
    df[column] = df[column].str.lower()
    #remove hyperlinks
    df[column] = df[column].apply(lambda x:re.split('https:\/\/.*',str(x))[0])
    #remove punctuations
    df[column] = df[column].str.replace('[^\w\s]'," ")
    #remove special characters
    df[column] = df[column].str.replace("\W"," ")
    #remove digits
    df[column] = df[column].str.replace("\d+"," ")
    #remove under scores
    df[column] = df[column].str.replace("_"," ")
    #remove stopwords
    df[column] = df[column].apply(lambda x: " ".join([i for i in x.split() 
                                                      if i not in (stop_words)]))
    return df

tweets = string_manipulation(tweet,"text")


print(tweets.head(20))

tweets['text'] = tweets['text'].str.split()


# Detect common phrases so that we may treat each one as its own word
phrases = gensim.models.phrases.Phrases(tweets['text'].tolist())
phraser = gensim.models.phrases.Phraser(phrases)
train_phrased = phraser[tweets['text'].tolist()]

# Gensim has support for multi-core systems
print(multiprocessing.cpu_count())

# I have no reason in mind to change the default word2vec parameters, so I will use the def
# aults
w2v = gensim.models.word2vec.Word2Vec(sentences=train_phrased,workers=4)


#traning our model
w2v.train(tweets['text'],total_examples=len(tweets['text']),epochs=10)

def analogy(x, y, a):
    b = w2v.wv.most_similar(positive=[a, y], negative=[x], topn=1)[0][0]
    return b, ' '.join([x,':',y,'::',a,':',b])
b, text = analogy('americans', 'america', 'french')
print(text)

# w1 = "america"
# print(" Most similar to america : "+str(w2v.wv.most_similar (positive=w1)))
# print(" Opposite to america : "+str(w2v.wv.most_similar (negative=["america"])))
# w1 = "loser"
# print(" Most similar to loser : "+str(w2v.wv.most_similar (positive=w1)))
# print(" Opposite to loser : "+str(w2v.wv.most_similar (negative=["loser"])))
# w1 = "france"
# print(" Most similar to France : "+str(w2v.wv.most_similar (positive=w1,topn=3)))
# w1 = "china"
# print(" Most similar to China: "+str(w2v.wv.most_similar (positive=w1,topn=3)))

# get everything pillow'related to stuff on the bed
#w1 = ["america","president","trump","donald","j","mr"]
#w2 = ['hillary']
#print(w2v.wv.most_similar (positive=w1,negative=w2,topn=10))

# cosine similarity between two unrelated words
#print("trump/Obama : "+str(w2v.wv.similarity(w1="trump",w2="obama")))
#print("trump/Hillary : "+str(w2v.wv.similarity(w1="trump",w2="hillary")))
#print("Obama/Hillary : "+str(w2v.wv.similarity(w1="obama",w2="hillary")))

# Which one is the odd one out in this list?
print("most odd america/great/trump/hillary : "+str(w2v.wv.doesnt_match(["america","great","trump","hillary"])))
