import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import spacy
import nltk
from sklearn.decomposition import LatentDirichletAllocation
import mglearn
import  re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('tweet.csv', header = 0,delimiter=',')


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
    #remove stopwords
    df[column] = df[column].apply(lambda x: " ".join([i for i in x.split()
                                                      if i not in (custom_stop_words)]))
    return df

tweets = string_manipulation(df,"text")

#Select only tweets
tweets = tweets['text']
print("nombre de Tweets : "+str(len(df)))

#Transform to list
vect = CountVectorizer()
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


# load spacy's English-language models
en_nlp = spacy.load('en')
# instantiate nltk's Porter stemmer
stemmer = nltk.stem.PorterStemmer()

# define function to compare lemmatization in spacy with stemming in nltk
def compare_normalization(doc):
    # tokenize document in spacy
    doc_spacy = en_nlp(doc)
    # print lemmas found by spacy
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

# Technicallity: we want to use the regexp based tokenizer
# that is used by CountVectorizer  and only use the lemmatization
# from SpaCy. To this end, we replace en_nlp.tokenizer (the SpaCy tokenizer)
# with the regexp based tokenization
import re
# regexp used in CountVectorizer:
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model
en_nlp = spacy.load('en', disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the preceding regexp
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))

# create a custom tokenizer using the SpaCy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]

# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

# transform text_train using CountVectorizer with lemmatization
X_train_lemma = lemma_vect.fit_transform(docs)
print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

# standard CountVectorizer for reference
vect = CountVectorizer(min_df=5).fit(docs)
X_train = vect.transform(docs)
print("X_train.shape: {}".format(X_train.shape))

vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(docs)

# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once


lda100 = LatentDirichletAllocation(100, learning_method="batch",
                                    max_iter=25, random_state=0)

print("lda.components_.shape: {}".format(lda100.components_.shape))

document_topics100 = lda100.fit_transform(X)

topics = np.array([7, 16, 24, 25, 28, 36, 37, 41, 45, 51, 53, 54, 63, 89, 97])

# for each topic (a row in the components_), sort the features (ascending).
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]

#get the feature names from the vectorizer:
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=20)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]
# two column bar chart:
for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 800)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
plt.savefig('d.png')