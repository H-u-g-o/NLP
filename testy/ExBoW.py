import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import  re

# Load the dataset
df = pd.read_csv('tweet.csv', header = 0,delimiter=',')

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

tweets = string_manipulation(df,"text")

#Select only tweets
tweets = df['text']

#Print column name
#for col in df.columns:
#    print(col)

# je transforme mon ensemble de tweet en une seule "phrase"
tweets_flatten = ''.join(tweets)
print("nombre de caractères : "+str(len(tweets_flatten)))

# je créer une liste où chaque élément est un mot
tweets_flatten_splitted = tweets_flatten.split()
print("nombre de mots : "+str(len(tweets_flatten_splitted)))

# je definie les stop words et les supprime de ma data
custom_stop_words = ["amp", "co", "http","th"]

tweets_flatten_splitted = [x for x in tweets_flatten_splitted if x not in custom_stop_words]

print(len(tweets_flatten_splitted))
# j'itere sur chaque mot (donc sur chaque element de ma list) et je calcul le nombre d'occurent EXACT de celui dans la phrase
wordfreq = []
for w in tweets_flatten_splitted:
    wordfreq.append(tweets_flatten_splitted.count(w))
print(len(wordfreq))

# je fais une association clef valeur pour le mot et l'occurence de celui-ci
result = dict((zip(tweets_flatten_splitted, wordfreq)))

# je "trie" bon dictionnaire par valeur afin d'avoir les mots ayant la plus grand occurence en premier
sorted_x = sorted(result.items(), key=lambda x: x[1])

print(sorted_x)

#wordcloud = WordCloud(width=1600, height=800, max_words=2000).generate(" ".join(flat_text))
wordcloud = WordCloud().generate_from_frequencies(result)

plt.figure(figsize=(30, 30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('cloud.png')