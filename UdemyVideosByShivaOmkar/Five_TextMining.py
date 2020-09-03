#%% Tokenizers
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Sentence tokenizer
text = "this's a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it is your turn."
sent_tokenize_list = sent_tokenize(text) # Uses PunktSentenceTokenizer trained on english. There are pretrained tokenizers for many languages.
len(sent_tokenize_list) #5
sent_tokenize_list

#Tokenizing text into words
word_tokenize(text) # Uses TreebankWordTokenizer

#Class work: Use the MWE (multi-word expression) tokenizer for "multi word - phrases"
#from nltk.tokenize.mwe import MWETokenizer
#%% Part-of-speech tagging
import nltk
text = "This is python training by Shiv for the Analytics team happening at Bangalore . he is felicitating and helping us ."
nltk.pos_tag(text.split())

#NLTK provides documentation for each tag
nltk.help.upenn_tagset("RB")
nltk.help.upenn_tagset("NN*")
nltk.help.upenn_tagset("JJ")
nltk.help.upenn_tagset("IN")

#%%TextBlob
from textblob import TextBlob

text = "Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models."
nlpblob = TextBlob(text)

#introduce various features of TextBlob

#1) Word Tokenization
nlpblob.words

#2) Sentence Tokenization
nlpblob.sentences

#3）Part-of-Speech Tagging
nlpblob.tags

#4) Noun Phrase Extraction
nlpblob.noun_phrases

#5) Sentiment Analysis
nlpblob.sentiment.polarity # polarity is a value between -1.0 and +1.0
nlpblob.sentiment.subjectivity # subjectivity between 0.0 and 1.0.
nlpblob.sentiment

#sentiment polarity
for sentence in nlpblob.sentences:
    print(sentence.sentiment.polarity)

#6) Word Singularize
nlpblob.words[138]
nlpblob.words[138].singularize()

#7) Word Pluralize
nlpblob.words[21]
nlpblob.words[21].pluralize()

#8) Words Lemmatization
#Words can be lemmatized by the lemmatize method, but notice that the TextBlog lemmatize method is
# inherited from NLTK Word Lemmatizer, and the default POS Tag is "n", if you want lemmatize other
#pos tag words, you need specify it:
nlpblob.words[138].pluralize().lemmatize()
nlpblob.words[21].pluralize().lemmatize()

#9）Spelling Correction
#TextBlob Spelling correction is based on Peter Norvig"s "How to Write a Spelling Corrector", which is
# implemented in the pattern library:
b = TextBlob("I havv good speling!")
b.correct()

#Word objects also have a spellcheck() method that returns a list of (word, confidence) tuples with spelling suggestions:

#9) Parsing: TextBlob parse method is based on pattern parser:
nlpblob.parse()

#10) Translation and Language Detection: By Google"s API:
#Detect
nlpblob.detect_language()

nlpblob.translate(to="hi")
nlpblob.translate(to="kn") # es fr
nlpblob.translate(to="fr") # es fr
nlpblob.translate(to="zh")

# Few more example. How to get keyword for any particular language
non_eng_blob = TextBlob("हिन्दी समाचार की आधिकारिक वेबसाइट. पढ़ें देश और दुनिया की ताजा ख़बरें")
non_eng_blob.detect_language()

non_eng_blob = TextBlob("ಮುಖ್ಯ ವಾರ್ತೆಗಳು ಜನಪ್ರಿಯ")
non_eng_blob.detect_language()
non_eng_blob.translate(to="en")
non_eng_blob.translate(to="hi")

#Class work: Try your native languages

#%%Pattern is a web mining module for the Python programming language.
#It has tools for data mining (Google, Twitter and Wikipedia API, a web crawler, a HTML DOM parser),
#natural language processing (part-of-speech taggers, n-gram search, sentiment analysis, WordNet), machine learning
# (vector space model, clustering, SVM), network analysis and canvas visualization.
#It is  text data mining tool which including crawler
# from pattern.en import *
#%% Stanford : Named-entity recognition (NER)
from nltk.tag.stanford import StanfordNERTagger
import nltk

# Locate nltk_data folder and provide path
st = StanfordNERTagger("D:\\nltk_data\\stanford-ner-2014-06-16\\classifiers\\english.all.3class.distsim.crf.ser.gz", "D:\\nltk_data\\stanford-ner-2014-06-16\\stanford-ner.jar")
text = "This is python training by Shiv for the Analytics team happening at Bangalore . he is felicitating and helping us ."
st.tag(nltk.word_tokenize(text))
# Notice the mistake in identification
#%% Stemming and Lemmatization
# NLTK provides several famous stemmers interfaces, such as Porter stemmer, Snowball Stemmer
#Lancaster Stemmer, .The aggressiveness continuum basically following along those same lines

#For Porter Stemmer, which is based on The Porter Stemming Algorithm, can be used like this

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

# Examples
porter_stemmer.stem("study")
porter_stemmer.stem("studies")
porter_stemmer.stem("studying")
porter_stemmer.stem("maximum")
porter_stemmer.stem("presumably")
porter_stemmer.stem("multiply")
porter_stemmer.stem("provision")
porter_stemmer.stem("owed")
porter_stemmer.stem("ear")
porter_stemmer.stem("saying")
porter_stemmer.stem("crying")
porter_stemmer.stem("string")
porter_stemmer.stem("meant")
porter_stemmer.stem("cement")

#For Snowball Stemmer, which is based on Snowball Stemming Algorithm, can be used in NLTK like this:
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
snowball_stemmer.stem("study")
snowball_stemmer.stem("studies")
snowball_stemmer.stem("studying")
snowball_stemmer.stem("maximum")
snowball_stemmer.stem("presumably")
snowball_stemmer.stem("multiply")
snowball_stemmer.stem("provision")
snowball_stemmer.stem("owed")
snowball_stemmer.stem("ear")
snowball_stemmer.stem("saying")
snowball_stemmer.stem("crying")
snowball_stemmer.stem("string")
snowball_stemmer.stem("meant")
snowball_stemmer.stem("cement")

#For Lancaster Stemmer, which is based on The Lancaster Stemming Algorithm
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
lancaster_stemmer.stem("study")
lancaster_stemmer.stem("studies")
lancaster_stemmer.stem("studying")
lancaster_stemmer.stem("maximum")
lancaster_stemmer.stem("presumably")
lancaster_stemmer.stem("presumably")
lancaster_stemmer.stem("multiply")
lancaster_stemmer.stem("provision")
lancaster_stemmer.stem("owed")
lancaster_stemmer.stem("ear")
lancaster_stemmer.stem("saying")
lancaster_stemmer.stem("crying")
lancaster_stemmer.stem("string")
lancaster_stemmer.stem("meant")
lancaster_stemmer.stem("cement")

#How to use Lemmatizer in NLTK. The NLTK Lemmatization method is based on WordNet's built-in morphy
# function. In NLTK, you can use it as the following:
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize("study")
wordnet_lemmatizer.lemmatize("studies")
wordnet_lemmatizer.lemmatize("studying")
wordnet_lemmatizer.lemmatize("dogs")
wordnet_lemmatizer.lemmatize("churches")
wordnet_lemmatizer.lemmatize("aardwolves")
wordnet_lemmatizer.lemmatize("abaci")
wordnet_lemmatizer.lemmatize("hardrock")
wordnet_lemmatizer.lemmatize("are")
wordnet_lemmatizer.lemmatize("is")
#You would note that the "are” and "is” lemmatize results are not "be”, that"s because the lemmatize
#method default pos argument is "n”. v(verb), a(adjective), r(adverb), n(noun).

#So you need specified the pos for the word like these:
wordnet_lemmatizer.lemmatize("is", pos="v")
wordnet_lemmatizer.lemmatize("are", pos="v")
#%% Word Cloud
import os
os.chdir("D:\Trainings\python")
from os import path
from wordcloud import  WordCloud, STOPWORDS #

# Read the whole text.
text = open(path.join('./data/constitution.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:

# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%% CW: Explore FastText (An NLP library by Facebook)
#http://feedproxy.google.com/~r/AnalyticsVidhya/~3/r-TzzESKAbQ/?utm_source=feedburner&utm_medium=email
