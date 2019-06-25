from nltk.corpus import stopwords
from sklearn.feature_extraction import text

#Getting stop words lists from NLTK & SKlearn
stop_words = stopwords.words("english")
print("NLTK stop words length: {} ".format(len(stop_words)))

my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
print("SKlearn stop words length: {} ".format(len(my_stop_words)))

#Combining both lists without duplicates
resultList = list(set(stop_words) | set(my_stop_words))
print("Combined (wo duplicates) length: {} ".format(len(resultList)))

#Removing usefull vocabulary
remove_list = ['cry','serious','further','less','please','detail','bill','bottom','sincere','against','together','nobody','amount', 'system', 'fire', 'below', 'due','empty','interest']
resultList = [i for i in resultList if i not in remove_list]
print("Removing usefull vocab : {} ".format(len(resultList)))

#Adding custom stopwords :
custom_stop_words = ["amp", "co", "http","th"]
resultList.extend(custom_stop_words)
print("Adding custom stop words : {} ".format(len(resultList)))

#Write Stop word list in .txt file :
with open('stopW.txt', 'w') as filehandle:
    for listitem in resultList:
        filehandle.write('%s\n' % listitem)