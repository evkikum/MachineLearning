#%% Word2Vec (semantic learning framework, KNN, cosine similarity)
from gensim.models import Word2Vec
from nltk.corpus import brown

#Train the model:
sentences = brown.sents()
len(sentences)
sentences[0]
model = Word2Vec(sentences, min_count=1)

#words most similar to mother
model.most_similar('mother')
model.most_similar('human')

#find the odd one out
text = 'breakfast cereal dinner lunch'
model.doesnt_match(text.split())
text = 'cat dog table'
model.doesnt_match(text.split())

#vector representation of word human
word_embedding = model['human']
len(word_embedding)
word_embedding
#%%Text Summarisation
from gensim.summarization import summarize

# https://en.wikipedia.org/wiki/Text_mining
text="""Text mining,[1] also referred to as text data mining, roughly equivalent to text analytics, is the process of deriving high-quality information from text. High-quality information is typically derived through the devising of patterns and trends through means such as statistical pattern learning. Text mining usually involves the process of structuring the input text (usually parsing, along with the addition of some derived linguistic features and the removal of others, and subsequent insertion into a database), deriving patterns within the structured data, and finally evaluation and interpretation of the output. 'High quality' in text mining usually refers to some combination of relevance, novelty, and interestingness. Typical text mining tasks include text categorization, text clustering, concept/entity extraction, production of granular taxonomies, sentiment analysis, document summarization, and entity relation modeling (i.e., learning relations between named entities).
Text analysis involves information retrieval, lexical analysis to study word frequency distributions, pattern recognition, tagging/annotation, information extraction, data mining techniques including link and association analysis, visualization, and predictive analytics. The overarching goal is, essentially, to turn text into data for analysis, via application of natural language processing (NLP) and analytical methods.
A typical application is to scan a set of documents written in a natural language and either model the document set for predictive classification purposes or populate a database or search index with the information extracted.
The term text analytics describes a set of linguistic, statistical, and machine learning techniques that model and structure the information content of textual sources for business intelligence, exploratory data analysis, research, or investigation.[2] The term is roughly synonymous with text mining; indeed, Ronen Feldman modified a 2000 description of "text mining"[3] in 2004 to describe "text analytics".[4] The latter term is now used more frequently in business settings while "text mining" is used in some of the earliest application areas, dating to the 1980s,[5] notably life-sciences research and government intelligence.
The term text analytics also describes that application of text analytics to respond to business problems, whether independently or in conjunction with query and analysis of fielded, numerical data. It is a truism that 80 percent of business-relevant information originates in unstructured form, primarily text.[6] These techniques and processes discover and present knowledge – facts, business rules, and relationships – that is otherwise locked in textual form, impenetrable to automated processing."""

summarise_text = summarize(text, ratio=0.2, word_count=None, split=False)
summarise_text
len(text), len(summarise_text)

#%% GloVe is another library for above

#%% Live sentiment Analysis http://nlp.stanford.edu:8080/sentiment/rntnDemo.html
# Class work: Play with above link. Make sure to put cursor on bubble to see the Analysis

#%% Keyword extraction is tasked with the automatic identification of terms that best describe the
#subject of a document.
#RAKE (Rapid Automatic Keyword Extraction). RAKE is a simple keyword extraction library which
#focuses on finding multi-word phrases containing frequent words. Its strengths are its simplicity
#and the ease of use , whereas its weaknesses are its limited accuracy, the parameter configuration
#requirement, and the fact that it throws away many valid phrases and doesn’t normalize candidates.

#A typical keyword extraction algorithm has three main components:
#a.Candidate selection: Here, we extract all possible words, phrases, terms or concepts (depending
#on the task) that can potentially be keywords.
#b.Properties calculation: For each candidate, we need to calculate properties that indicate that
#it may be a keyword. For example, a candidate appearing in the title of a book is a likely keyword.
#c. Scoring and selecting keywords: All candidates can be scored by either combining the properties into a
# formula, or using a machine learning technique to determine probability of a candidate being a keyword.
# A score or probability threshold, or a limit on the number of keywords is then used to select the final
# set of keywords.
#RAKE follow the three steps strictly

import os
import RAKE
import operator

os.chdir("D:\\trainings\\python")

#Create object and provide stop list
rake_object = RAKE.Rake("SmartStoplist.txt") # , 3, 3, 1
#Each word has at least 3 characters
#Each phrase has at most 3 words
#Each keyword appears in the text at least 1 times

text = "Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models. In this course you will study mathematical and computational models of language, and the application of these models to key problems in natural language processing. The course has a focus on machine learning methods, which are widely used in modern NLP systems: we will cover formalisms such as hidden Markov models, probabilistic context-free grammars, log-linear models, and statistical models for machine translation. The curriculum closely follows a course currently taught by Professor Collins at Columbia University, and previously taught at MIT."
keywords = rake_object.run(text)
print(keywords) # keyword and its score

# Each keyword appears in the text at least 2 times
#rake_object = RAKE.Rake("SmartStoplist.txt", 3, 3, 2)
#keywords = rake_object.run(text)
#print(keywords)

#The key points for RAKE is the parameters setting, and RAKE provides a method to select a proper
#parameters based on the train data. As the document summarize, RAKE is very easy to use to getting
#start keyword extraction, but seems lack something:

#%% Finding Important Words in Text Using TF-IDF
#TF-IDF stands for "Term Frequency, Inverse Document Frequency". It is a way to score the importance of
#words (or "terms") in a document based on how frequently they appear across multiple documents.
#If a word appears frequently in a document, it's important. Give the word a high score.
#But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
#Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.

# TF * IDF = [ (Number of times term t appears in a document) / (Total number of terms in the document) ]
#            * log10(Total number of documents / Number of documents with term t in it)

import math
from textblob import TextBlob as tb

#Few helper functions
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

# The more common a word is, the lower its idf
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

#Temporary data
document1 = tb("""Technology outsourcing companies are seeking fewer work visas from the United States government as evident from the declining number of petitions filed this year, according to data disclosed by the US Citizenship and Immigration Services.""")

document2 = tb("""The administration, in turn, has approved less than 59 per cent of H-1B visa applications as sentiment against foreign workers turns negative in the largest market for tech outsourcing.
This year, the American government received over 3,36,000 petitions for H-1B visas, both extensions and new applications, and had approved about 1,97,129 at the end of June.""")

document3 = tb("""They have become stricter and are asking for more supporting documentation," said an executive at an Indian IT outsourcing firm who declined to be identified. He estimates the final number of visa applications may not be "dramatically lower" but said "Indian IT companies have already started to reduce their visa requests. Indian IT companies, one of the biggest users of the work visa, have been cutting back as protectionist rhetoric grows in their major market. Wipro went so far as to name US President Donald Trump has a risk factor.
""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

#The tf-idf value increases proportionally to the number of times a word appears in the document,
#but is often offset by the frequency of the word in the corpus, which helps to adjust for the fact
#that some words appear more frequently in general.
