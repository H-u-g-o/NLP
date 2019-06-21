import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import  re
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('tweet.csv', header = 0,delimiter=',')

#Print column name
#for col in df.columns:
#    print(col)

#Define stopwords
stop_words = stopwords.words("english")
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
    return df

tweets = string_manipulation(df,"text")

#Select only tweets
tweets = tweets['text']
print("nombre de Tweets : "+str(len(df)))

#Transform to list

vect =  CountVectorizer()
vect.fit(tweets)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
#print("Vocabulary content: \n {}".format(vect.vocabulary_))

bag_of_words = vect.transform(tweets)
print("bag_of_words: {}".format(repr(bag_of_words)))

features_names = vect.get_feature_names()
print("Number of features: {}".format(len(features_names)))
print("Every 1000 features: {}".format(features_names[::1000]))

#TF-idf
#create a vocabulary of words,
#ignore words that appear in 85% of documents,
#limit our vocabulary size to 10,000
docs=tweets.tolist()
cv=CountVectorizer(max_df=0.85,stop_words=custom_stop_words,max_features=10000)
word_count_vector=cv.fit_transform(docs)
print(word_count_vector.shape)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# you only needs to do this once
feature_names=cv.get_feature_names()

doc=str(docs)

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))


#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)

# now print the results
print("\n===Keywords with highest Tfidf===")
for k in keywords:
    print(k,keywords[k])
