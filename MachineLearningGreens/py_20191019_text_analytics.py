## Text Analytics, Natural Language Processing (NLP)
  ## Information retrieval, Computational linguistics

import pandas as pd
import numpy as np
import nltk # natural language toolkit
#nltk.download()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # TF, TFIDF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob


################# Text Processing ########################################

######### Tokenization
## Splitting a text into individual tokens (words)

txt = "this is python class"
txt_words = txt.split(" ")

## Word Tokenizer
txt_words2 = nltk.tokenize.word_tokenize(txt)

## Tweet Tokenizer
twt = "@PMOffice: Chinese president Xi Jinping has reached Mamallapuram with 200 delegates. @narendramodi is welcoming him there"
twt_words = twt.split()
twt_tokenize1 = nltk.tokenize.TweetTokenizer().tokenize(twt)
twt_tokenize2 = nltk.tokenize.TweetTokenizer(strip_handles = True,
                    preserve_case = False).tokenize(twt)

## Regex Tokenizer
# https://docs.python.org/3/howto/regex.html
text2 = "this is great-going!, however is there a way to improve sales 10 times or 100 times?"
text2_tokenize1 = nltk.tokenize.RegexpTokenizer("\w+").tokenize(text2) # extracting only alpha numeric
text2_tokenize2 = nltk.tokenize.RegexpTokenizer("\d+").tokenize(text2) # extract numbers
text2_tokenize3 = nltk.tokenize.RegexpTokenizer("\D+").tokenize(text2)

### Stop word corpus
nltk_stopwords_corpus = nltk.corpus.stopwords
nltk_stopwords_corpus.fileids()
eng_stopwords = nltk.corpus.stopwords.words("english")
german_stopwords = nltk.corpus.stopwords.words("german")

text2_stopwords_removed = [i for i in text2_tokenize1 if i not in eng_stopwords]

#### Stemming
nltk.stem.PorterStemmer().stem("oranges")
nltk.stem.PorterStemmer().stem("orange")

nltk.stem.PorterStemmer().stem("winning")
nltk.stem.PorterStemmer().stem("wins")

nltk.stem.PorterStemmer().stem("winner")
nltk.stem.PorterStemmer().stem("winners")

############## Text Classification (Sentiment ANalysis) #################################

## Training tweets
tr_tweets = ['I love this car',
'This view is amazing',
'I feel great this morning',
'I am so excited about the concert',
'He is my best friend',
'I do not like this car',
'This view is horrible',
'I feel tired this morning',
'I am not looking forward to the concert',
'He is my enemy'] #IDVs
tr_sentiment = ["positive","positive","positive","positive","positive",
                "negative","negative","negative","negative","negative"] # DV


## Testing tweets
te_tweets = ['I feel happy this morning', 
'Larry is my friend',
'I do not like that man',
'This view is horrible',
'The house is not great',
'Your song is annoying']
te_sentiment = ["positive","positive","negative","negative","negative","negative"]

########## Feature Extraction

# Below line will throw error because no algorithm works on text data
# twt_logit = LogisticRegression().fit(tr_tweets,tr_sentiment)

# Feature Set 1: Retaining the case and retaining all words
twt_vectorizer1 = CountVectorizer(lowercase = False, stop_words = None)
twt_tr_vector1 = twt_vectorizer1.fit(tr_tweets)
twt_tr_vector1_feat = twt_tr_vector1.get_feature_names() # 29 terms
tr_tweets_transformed1 = twt_tr_vector1.transform(tr_tweets) # Transforming training data
print(tr_tweets_transformed1)
# converting from sparse matrix to raw matrix
tr_tweets_transformed1_rawmat = pd.DataFrame(tr_tweets_transformed1.toarray(),
                                columns = twt_tr_vector1_feat)
te_tweets_transformed1 = twt_tr_vector1.transform(te_tweets) # Transforming test data


# Feature Set 2: Removed stopwords and converted all to lower case
twt_vectorizer2 = CountVectorizer(lowercase = True, stop_words = "english")
twt_tr_vector2 = twt_vectorizer2.fit(tr_tweets)
twt_tr_vector2_feat = twt_tr_vector2.get_feature_names() # 17 terms
tr_tweets_transformed2 = twt_tr_vector2.transform(tr_tweets)
print(tr_tweets_transformed2)
te_tweets_transformed2 = twt_tr_vector2.transform(te_tweets) # Transforming test data

# Feature Set 3: Included bigram as well
twt_vectorizer3 = CountVectorizer(lowercase = True, stop_words = "english",
                                  ngram_range = (1,2))
twt_tr_vector3 = twt_vectorizer3.fit(tr_tweets)
twt_tr_vector3_feat = twt_tr_vector3.get_feature_names() # 29 terms
tr_tweets_transformed3 = twt_tr_vector3.transform(tr_tweets)
te_tweets_transformed3 = twt_tr_vector3.transform(te_tweets) # Transforming test data

# Feature Set 4: TFIDF
twt_vectorizer4 = TfidfVectorizer(lowercase = True, stop_words = "english")
twt_tr_vector4 = twt_vectorizer4.fit(tr_tweets)
twt_tr_vector4_feat = twt_tr_vector4.get_feature_names() # 17 terms
tr_tweets_transformed4 = twt_tr_vector4.transform(tr_tweets)
# converting from sparse matrix to raw matrix
tr_tweets_transformed4_rawmat = pd.DataFrame(tr_tweets_transformed4.toarray(),
                                columns = twt_tr_vector4_feat)
te_tweets_transformed4 = twt_tr_vector4.transform(te_tweets) # Transforming test data


########## Model Building

# Feature Set 1, Decision Tree
twt_model1 = DecisionTreeClassifier().fit(tr_tweets_transformed1,tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model1) #100% accuracy
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model1) #83% accuracy

# Feature Set 1, Logistic Regression
twt_model2 = LogisticRegression().fit(tr_tweets_transformed1,tr_sentiment)
sent_pre_tr_model2 = twt_model2.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model2) #100% accuracy
sent_pre_te_model2 = twt_model2.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model2) #83% accuracy

# Feature Set 2, Decision Tree
twt_model3 = DecisionTreeClassifier().fit(tr_tweets_transformed2,tr_sentiment)
sent_pre_tr_model3 = twt_model3.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model3) #100% accuracy
sent_pre_te_model3 = twt_model3.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model3) #60% accuracy

# Feature Set 3, Decision Tree
twt_model4 = DecisionTreeClassifier().fit(tr_tweets_transformed3,tr_sentiment)
sent_pre_tr_model4 = twt_model4.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model4) #100% accuracy
sent_pre_te_model4 = twt_model4.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model4) #66% accuracy

# Feature Set 4, Decision Tree
twt_model5 = DecisionTreeClassifier().fit(tr_tweets_transformed4,tr_sentiment)
sent_pre_tr_model5 = twt_model5.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model4) #100% accuracy
sent_pre_te_model5 = twt_model5.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model5) #66% accuracy


##########################################################################

## Happy Train
f = open("data/happy.txt","r",encoding='utf8')
happy_train = f.readlines()
f.close()
## Happy Test
f = open("data/happy_test.txt","r",encoding='utf8')
happy_test = f.readlines()
f.close()
## Sad Train
f = open("data/sad.txt","r",encoding='utf8')
sad_train = f.readlines()
f.close()
## Sad Test
f = open("data/sad_test.txt","r",encoding='utf8')
sad_test = f.readlines()
f.close()

## Creating Training and Test data
tr_tweets = happy_train + sad_train
tr_sentiment = ["happy"]*80 + ["sad"]*80

te_tweets = happy_test + sad_test
te_sentiment = ["happy"]*10 + ["sad"]*10

########## Feature Extraction
# Feature Set 1: Retaining the case and retaining all words
twt_vectorizer1 = CountVectorizer(lowercase = False, stop_words = None)
twt_tr_vector1 = twt_vectorizer1.fit(tr_tweets)
twt_tr_vector1_feat = twt_tr_vector1.get_feature_names() # 841 terms
tr_tweets_transformed1 = twt_tr_vector1.transform(tr_tweets) # Transforming training data
print(tr_tweets_transformed1)
# converting from sparse matrix to raw matrix
tr_tweets_transformed1_rawmat = pd.DataFrame(tr_tweets_transformed1.toarray(),
                                columns = twt_tr_vector1_feat)
te_tweets_transformed1 = twt_tr_vector1.transform(te_tweets) # Transforming test data


# Feature Set 2: Removed stopwords and converted all to lower case
twt_vectorizer2 = CountVectorizer(lowercase = True, stop_words = "english",
                                  max_features = 100)
twt_tr_vector2 = twt_vectorizer2.fit(tr_tweets)
twt_tr_vector2_feat = twt_tr_vector2.get_feature_names() # 100 terms
tr_tweets_transformed2 = twt_tr_vector2.transform(tr_tweets)
te_tweets_transformed2 = twt_tr_vector2.transform(te_tweets) # Transforming test data

# Feature Set 3: Included bigram as well
twt_vectorizer3 = CountVectorizer(lowercase = True, stop_words = "english",
                                  ngram_range = (1,2),max_features = 100)
twt_tr_vector3 = twt_vectorizer3.fit(tr_tweets)
twt_tr_vector3_feat = twt_tr_vector3.get_feature_names()
tr_tweets_transformed3 = twt_tr_vector3.transform(tr_tweets)
te_tweets_transformed3 = twt_tr_vector3.transform(te_tweets) # Transforming test data

# Feature Set 4: TFIDF
twt_vectorizer4 = TfidfVectorizer(lowercase = True, stop_words = "english",
                                  max_features = 100)
twt_tr_vector4 = twt_vectorizer4.fit(tr_tweets)
twt_tr_vector4_feat = twt_tr_vector4.get_feature_names()
tr_tweets_transformed4 = twt_tr_vector4.transform(tr_tweets)
# converting from sparse matrix to raw matrix
tr_tweets_transformed4_rawmat = pd.DataFrame(tr_tweets_transformed4.toarray(),
                                columns = twt_tr_vector4_feat)
te_tweets_transformed4 = twt_tr_vector4.transform(te_tweets) # Transforming test data

########## Model Building

# Feature Set 1, Decision Tree
twt_model1 = DecisionTreeClassifier().fit(tr_tweets_transformed1,tr_sentiment)
sent_pre_tr_model1 = twt_model1.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model1) #99.3% accuracy
sent_pre_te_model1 = twt_model1.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model1) #95% accuracy

# Feature Set 1, Logistic Regression
twt_model2 = LogisticRegression().fit(tr_tweets_transformed1,tr_sentiment)
sent_pre_tr_model2 = twt_model2.predict(tr_tweets_transformed1)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model2) #99.3% accuracy
sent_pre_te_model2 = twt_model2.predict(te_tweets_transformed1)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model2) #95% accuracy

# Feature Set 2, Decision Tree
twt_model3 = DecisionTreeClassifier().fit(tr_tweets_transformed2,tr_sentiment)
sent_pre_tr_model3 = twt_model3.predict(tr_tweets_transformed2)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model3) #99.3% accuracy
sent_pre_te_model3 = twt_model3.predict(te_tweets_transformed2)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model3) #95% accuracy

# Feature Set 3, Decision Tree
twt_model4 = DecisionTreeClassifier().fit(tr_tweets_transformed3,tr_sentiment)
sent_pre_tr_model4 = twt_model4.predict(tr_tweets_transformed3)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model4) #99.3% accuracy
sent_pre_te_model4 = twt_model4.predict(te_tweets_transformed3)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model4) #95% accuracy

# Feature Set 4, Decision Tree
twt_model5 = DecisionTreeClassifier().fit(tr_tweets_transformed4,tr_sentiment)
sent_pre_tr_model5 = twt_model5.predict(tr_tweets_transformed4)
pd.crosstab(np.array(tr_sentiment),sent_pre_tr_model4) #99.3% accuracy
sent_pre_te_model5 = twt_model5.predict(te_tweets_transformed4)
pd.crosstab(np.array(te_sentiment),sent_pre_te_model5) #95% accuracy

#####################################################################
### Pretrained model in Text Blob
# Returns a polarity between -1 to +1
  # -1: Negative
  # +1: Positive
  # 0: Neutral

TextBlob("Your song is annoying").polarity
TextBlob("Larry is my friend").polarity
TextBlob("I feel happy this morning").polarity

pre_sent = []
for i in te_tweets:
    if (TextBlob(i).polarity > 0.3):
        pre_sent.append("happy")
    elif (TextBlob(i).polarity < -0.3):
        pre_sent.append("sad")
    else:
        pre_sent.append("neutral")

pd.crosstab(np.array(te_sentiment),np.array(pre_sent))
11/20 # 55% accuracy
            


