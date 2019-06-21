
# # Defining the structure of our word2vec model 

# # Size is the dimentionality feature of the model 
# model_1 = Word2Vec(size=300, min_count=1)
# #Feeding Our coupus 
# model_1.build_vocab(tweets['text'])
# #Lenth of the courpus 
# total_examples = model_1.corpus_count
# #traning our model
# model_1.train(tweets['text'], total_examples=total_examples,epochs=10)

# # fit a 2d PCA model to the vectors

# # X holds the vectors of n dimentions for each word in our vocab
# X = model_1[model_1.wv.vocab]

# # We are reducing the n dimentions to 2d
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)


# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model_1.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.savefig("fifig.png")