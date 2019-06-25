import pandas as pd
from nltk.corpus import stopwords
import  re
from sklearn.feature_extraction.text import CountVectorizer


# Load the dataset
df = pd.read_csv('tweet.csv', header = 0,delimiter=',')


#Define stopwords
stop_words = stopwords.words("english")
print(stop_words)

custom_stop_words = ["amp", "co", "http","th"]

#function to remove special characters , punctions ,stop words ,
#digits ,hyperlinks and case conversion
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
    #remove stopwords
    df[column] = df[column].apply(lambda x: " ".join([i for i in x.split()
                                                      if i not in (custom_stop_words)]))
    return df

#Cleaning our data applying the function
tweets = string_manipulation(df,"text")

#Select only tweets and transform to list
tweets = tweets['text'].tolist()
print("Total number of Tweets: {} ".format(len(df)))

#Create a vocabulary of words : each column represents a word in the vocabulary and each row
# represents the document in our dataset where the values are the word counts
#ignore words that appear in 85% of documents
#eliminate stop words ' : 176 words, lowercase proofing : no change
#encoding : ‘utf-8’ by default.
#max_df is used for removing terms that appear too frequently : in more than 85% of the documents
#min_df=0.01 ignore terms that appear in less than 1% of the documents" : only 120 words left.
cv = CountVectorizer(max_df=0.85, lowercase=True)
word_count_vector=cv.fit_transform(tweets)

print("Vocabulary size: {} unique words".format(len(cv.vocabulary_)))
#print("Vocabulary content: \n {}".format(vect.vocabulary_))

# bag_of_words = cv.transform(tweets)
# print("bag_of_words: {}".format(repr(bag_of_words)))

# features_names = vect.get_feature_names()
# print("Number of features: {}".format(len(features_names)))
# print("Every 1000 features: {}".format(features_names[::1000]))