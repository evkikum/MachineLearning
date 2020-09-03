#%% Sentiment analysis (also known as opinion mining)
import nltk
from nltk.corpus import movie_reviews
from random import shuffle

documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]

shuffle(documents)
print(documents[0])

# The total number of movie reviews documents in nltk is 2000
len(documents) # 2000

# Construct a list of the 2,000 most frequent words in the overall corpus
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:1000]

# Define a feature extractor that simply checks whether each of these words is present in a given document.
def document_features(document):
   document_words = set(document)
   features = {}
   for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
   return features

# Test above function
print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) # Part of library and no need of explicit download

# Generate the feature sets for the movie review documents one by one
featuresets = [(document_features(d), c) for (d, c) in documents]

# Define the train set (1900 documents) and test set (100 documents)
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a naive bayes classifier with train set by nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Get the accuracy of the naive bayes classifier with test set
print(nltk.classify.accuracy(classifier, test_set)) #0.81

# Debug info: show top n most informative features
classifier.show_most_informative_features(10)

#Based on the top-2000 word features, we can train a Maximum entropy classifier model with NLTK
# for 50 iteration, it will take 3 hours and hence practice with 5 iteration
maxent_classifier = nltk.MaxentClassifier.train(train_set, max_iter = 50)# #MEGAM(algorithms:External Libraries)."megam" need some external configuration
print(nltk.classify.accuracy(maxent_classifier, test_set)) # 100:0.89, 10: .52

maxent_classifier.show_most_informative_features(10)

#It seems that the maxent classifier has the better classifier result on the test set. Let’s classify a
#test text with the Naive Bayes Classifier and Maxent Classifier:

test_text = "I love this movie, very interesting"
test_set = document_features(test_text.split())
test_set

# Naivebayes classifier result
print(classifier.classify(test_set))

# Maxent Classifier result
print(maxent_classifier.classify(test_set))

# Let's see the probability result
prob_result = classifier.prob_classify(test_set)
prob_result.prob("neg")
prob_result.prob("pos")

# Maxent classifier probability result
print(maxent_classifier.classify(test_set)) # pos
prob_result = maxent_classifier.prob_classify(test_set)
prob_result.prob("pos")
prob_result.prob("neg")

#Till now, we just used the top-n word features, and for this sentiment analysis machine learning problem,
#add more features may be get better result. So we redesign the word features:
def bag_of_words(words):
   return dict([(word, True) for word in words])

data_sets = [(bag_of_words(d), c) for (d, c) in documents]
len(data_sets) #44 2000

train_set, test_set = data_sets[100:], data_sets[:100]

bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(bayes_classifier, test_set)) #0.8
bayes_classifier.show_most_informative_features(10)

maxent_bg_classifier = nltk.MaxentClassifier.train(train_set) # , "megam"
print(nltk.classify.accuracy(maxent_bg_classifier, test_set)) #0.89

maxent_bg_classifier.show_most_informative_features(10)

#Now we can test the bigrams feature in the classifier model:
from nltk import ngrams
def bag_of_ngrams(words, n=2):
    ngs = [ng for ng in iter(ngrams(words, n))]
    return bag_of_words(ngs)

data_sets = [(bag_of_ngrams(d), c) for (d, c) in documents]
train_set, test_set = data_sets[100:], data_sets[:100]

nb_bi_classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(nb_bi_classifier, test_set)) #0.83

nb_bi_classifier.show_most_informative_features(10)

maxent_bi_classifier = nltk.MaxentClassifier.train(train_set) # , "megam"
print(nltk.classify.accuracy(maxent_bi_classifier, test_set)) # 0.9

maxent_bi_classifier.show_most_informative_features(10)

#And again, we can use the words feature and ngrams (bigrams) feature together:
def bag_of_all(words, n=2):
  all_features = bag_of_words(words)
  ngram_features = bag_of_ngrams(words, n=n)
  all_features.update(ngram_features)
  return all_features

data_sets = [(bag_of_all(d), c) for (d, c) in documents]
train_set, test_set = data_sets[100:], data_sets[:100]
nb_all_classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(nb_all_classifier, test_set)) #0.83

nb_all_classifier.show_most_informative_features(10)

maxent_all_classifier = nltk.MaxentClassifier.train(train_set) #, "megam"
maxent_all_classifier.show_most_informative_features(10)

print(nltk.classify.accuracy(maxent_all_classifier, test_set)) # 0.91
