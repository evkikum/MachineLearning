#%%Installing NLTK Data
#install NLTK Data which include a lot of corpora, grammars, models and etc. You can find the complete nltk data list
#here: http://nltk.org/nltk_data/ . If any issue then download directly from https://github.com/nltk/nltk_data
# change download folder to in D:\nltk_data or C: or E: (Windows), /usr/local/share/nltk_data (Mac), /usr/share/nltk_data (Unix)
import nltk
nltk.download()

#%% Test NLTK
#1) Test Brown Corpus:
from nltk.corpus import brown

brown.words()[0:10]
brown.tagged_words()[0:10]
len(brown.words()) #1161192
dir(brown)

#2) Test NLTK Book Resources:
from nltk.book import * # Will import all corpus from Text1-9

dir(text1)
len(text1) # 260819

#3) Sent Tokenize(sentence boundary detection, sentence segmentation), Word Tokenize and Pos Tagging:
from nltk import sent_tokenize, word_tokenize, pos_tag
text = "Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI."
sents = sent_tokenize(text)
sents
len(sents) # 7

# Each word as token
tokens = word_tokenize(text)
tokens
len(tokens) # 161

tagged_tokens = pos_tag(tokens)
tagged_tokens
