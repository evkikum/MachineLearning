#%%
#Data download
#https://gist.githubusercontent.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235/raw/1d2261e2276cbb0257a2ed6e2f1f4320464c7c07/corpus

# Steps
#1. Dataset Preparation
#2. Feature Engineering
#3. Model Training
#4. Improve Performance of Text Classifier

#%% libraries for dataset preparation, feature engineering, model training
import os
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#import seaborn as sns

# Working directory
os.chdir("D:\Trainings\python")
exec(open(os.path.abspath('CommonUtils.py')).read())

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%% 1. Dataset preparation

# load the dataset
data = open('./data/text_classificatios_label_text.txt').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(' '.join(word for word in content[1:]))

del(data)

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
trainDF.head(2)

del(texts, labels)

# first view
trainDF.shape # (10000, 2)
trainDF.head(2)

#Split (75%) the dataset into training and validation. Also, encode target column so that it can
#be used in machine learning models

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
#del(trainDF)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#2. Feature Engineering
#raw text data will be transformed

#2.1 Count Vectors as features
#2.2 TF-IDF Vectors as features

#Word level
#N-Gram level
#Character level
#2.3 Word Embeddings as features
#2.4 Text / NLP based features
#2.5 Topic Models as features

#2.1 Count Vectors as features
#Count Vector is a matrix notation of the dataset in which every row represents a document from
#the corpus, every column represents a term from the corpus, and every cell represents the frequency
#count of a particular term in a particular document.

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text']) # TBD

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

#2.2 TF-IDF Vectors as features
#TF-IDF score represents the relative importance of a term in the document and the entire corpus.
#TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF),
#the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number
#of the documents in the corpus divided by the number of documents where the specific term appears.
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

#TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams)
#a. Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents
#b. N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams
#c. Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

#2.3 Word Embeddings
#A word embedding is a form of representing words and documents using a dense vector representation.
#The position of a word within the vector space is learned from text and is based on the words
#that surround the word when it is used. Word embeddings can be trained using the input corpus
#itself or can be generated using pre-trained word embeddings such as Glove, FastText, and Word2Vec.
#Any one of them can be downloaded and used as transfer learning. One can read more about word
#embeddings here.

#Following snnipet shows how to use pre-trained word embeddings in the model. There are four
#essential steps:
#Loading the pretrained word embeddings
#Creating a tokenizer object
#Transforming text documents to sequence of tokens and pad them
#Create a mapping of token and their respective embeddings
#You can download the pre-trained word embeddings from here

# load the pre-trained word-embedding vectors
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
train_seq_x[0,:]

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#2.4 Text / NLP based features
#A number of extra text based features can also be created which sometimes are helpful for improving
# text classification models. Some examples are:
#Word Count of the documents – total number of words in the documents
#Character Count of the documents – total number of characters in the documents
#Average Word Density of the documents – average length of the words used in the documents
#Puncutation Count in the Complete Essay – total number of punctuation marks in the documents
#Upper Case Count in the Complete Essay – total number of upper count words in the documents
#Title Word Count in the Complete Essay – total number of proper case (title) words in the documents
#Frequency distribution of Part of Speech Tags:
#Noun Count
#Verb Count
#Adjective Count
#Adverb Count
#Pronoun Count
#These features are highly experimental ones and should be used according to the problem statement only.

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))
trainDF.head(2)

#2.5 Topic Models as features
#Topic Modelling is a technique to identify the groups of words (called a topic)
#from a collection of documents that contains best information in the collection.
#I have used Latent Dirichlet Allocation for generating Topic Modelling Features.
#LDA is an iterative model which starts from a fixed number of topics. Each topic
#is represented as a distribution over words, and each document is then represented
#as a distribution over topics. Although the tokens themselves are meaningless,
#the probability distributions over words provided by the topics provide a sense
#of the different ideas contained in the documents. One can read more about
#topic modelling here

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

topic_summaries

#3. Model Building
#The final step in the text classification framework is to train a classifier using the features
#created in the previous step. There are many different choices of machine learning models which
#can be used to train a final model. We will implement following different classifiers for this purpose:

#Naive Bayes Classifier
#Linear Classifier
#Support Vector Machine
#Bagging Models
#Boosting Models
#Shallow Neural Networks
#Deep Neural Networks
#Convolutional Neural Network (CNN)
#Long Short Term Modelr (LSTM)
#Gated Recurrent Unit (GRU)
#Bidirectional RNN
#Recurrent Convolutional Neural Network (RCNN)
#Other Variants of Deep Neural Networks
#Lets implement these models and understand their details. The following function is a utility
# function which can be used to train a model. It accepts the classifier, feature_vector of
# training data, labels of training data and feature vectors of valid data as inputs. Using
# these inputs, the model is trained and accuracy score is computed.

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    if is_neural_net:
        classifier.fit(feature_vector_train, label, epochs = 10)
    else:
        classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    accuracy = metrics.accuracy_score(predictions, valid_y)
    print("Accuracy: " + str(accuracy))
    return accuracy

#3.1 Naive Bayes
#Implementing a naive bayes model using sklearn implementation with different features
#Naive Bayes is a classification technique based on Bayes’ Theorem with an assumption of
#independence among predictors. A Naive Bayes classifier assumes that the presence of a
#particular feature in a class is unrelated to the presence of any other feature here .

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: " + str(accuracy)) # 0.8224

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: " + str(accuracy)) # 0.836

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: " + str(accuracy)) # 0.8312

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: " + str(accuracy)) # 0.8024

#3.2 Linear Classifier: Implementing a Linear Classifier (Logistic Regression)
#Logistic regression measures the relationship between the categorical dependent variable and
#one or more independent variables by estimating probabilities using a logistic/sigmoid function.

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
# 0.8696

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
# 0.8788

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# 0.834

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
# 0.834

#3.3 Implementing a SVM Model
#Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for
#both classification or regression challenges. The model extracts a best possible hyper-plane /
#line that segregates the two classes.

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# 0.5064

#3.4 Bagging Model: Implementing a Random Forest Model
#Random Forest models are a type of ensemble models, particularly bagging models. They are part
#of the tree based model family

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
# 0.7364

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
# 0.7728

#3.5 Boosting Model: Implementing Xtereme Gradient Boosting Model
#Boosting models are another type of ensemble models part of tree based models. Boosting is a
#machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in
#supervised learning, and a family of machine learning algorithms that convert weak learners
#to strong ones. A weak learner is defined to be a classifier that is only slightly correlated
#with the true classification (it can label examples better than random guessing)

# Extereme Gradient Boosting on Count Vectors. 'tocsc' means Compressed Sparse Column format
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
# 0.808

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
#0.8104

# Extereme Gradient Boosting on Character Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
# 0.8108

#3.6 Shallow Neural Networks: A neural network is a mathematical model that is designed to behave
#similar to biological neurons and nervous system. These models are used to recognize complex
#patterns and relationships that exists within a labelled data. A shallow neural network contains
#mainly three types of layers – input layer, hidden layer, and output layer.

def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
# 0.5064

#3.7 Deep Neural Networks: Deep Neural Networks are more complex neural networks in which the
#hidden layers performs much more complex operations than simple sigmoid or relu activations.
#Different types of deep learning models can be applied in text classification problems.

#3.7.1 Convolutional Neural Network: In Convolutional neural networks, convolutions over the
#input layer are used to compute the output. This results in local connections, where each
#region of the input is connected to a neuron in the output. Each layer applies different
#filters and combines their results.

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)

#3.7.2 Recurrent Neural Network – LSTM: Unlike Feed-forward neural networks in which activation
#outputs are propagated only in one direction, the activation outputs from neurons propagate
#in both directions (from inputs to outputs and from outputs to inputs) in Recurrent Neural
#Networks. This creates loops in the neural network architecture which acts as a ‘memory state’
#of the neurons. This state allows the neurons an ability to remember what have been learned so far.

#The memory state in RNNs gives an advantage over traditional neural networks but a problem
#called Vanishing Gradient is associated with them. In this problem, while learning with a
#large number of layers, it becomes really hard for the network to learn and tune the parameters
#of the earlier layers. To address this problem, A new type of RNNs called LSTMs (Long Short
#Term Memory) Models have been developed.

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)

#3.7.3 Recurrent Neural Network – GRU: Gated Recurrent Units are another form of recurrent
#neural networks. Lets add a layer of GRU instead of LSTM in our network.

def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
#RNN-GRU, Word Embeddings 0.5124

#3.7.4 Bidirectional RNN: RNN layers can be wrapped in Bidirectional layers as well. Lets wrap
# our GRU layer in bidirectional layer.
def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
#RNN-Bidirectional, Word Embeddings 0.5124

#3.7.5 Recurrent Convolutional Neural Network: Once the essential architectures have been tried
#out, one can try different variants of these layers such as recurrent convolutional neural
#network. Another variants can be:


#1. Text Cleaning : text cleaning can help to reducue the noise present in text data in the form of stopwords, punctuations marks, suffix variations etc. This article can help to understand how to implement text classification in detail.
#2. Hstacking Text / NLP features with text feature vectors : In the feature engineering section, we generated a number of different feature vectros, combining them together can help to improve the accuracy of the classifier.
#3. Hyperparamter Tuning in modelling : Tuning the paramters is an important step, a number of parameters such as tree length, leafs, network paramters etc can be fine tuned to get a best fit model.
#4. Ensemble Models : Stacking different models and blending their outputs can help to further improve the results. Read more about ensemble models here
