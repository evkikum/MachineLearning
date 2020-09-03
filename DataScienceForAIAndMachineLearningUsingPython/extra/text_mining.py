#Use the MWE (multi-word expression) tokenizer for "multi word - phrases" from nltk.tokenize.mwe import MWETokenizer
# A ``MWETokenizer`` takes a string which has already been divided into tokens and
#retokenizes it, merging multi-word expressions into single tokens, using a lexicon of MWEs:
from nltk.tokenize.mwe import MWETokenizer

text = "this's a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it is your turn."
mwe_tokenizer = MWETokenizer([('a', 'sent'), ('is', 'sent'), ('it', 'is', 'your')])
mwe_tokenizer.add_mwe(('this', 'is'))

mwe_tokenizer.tokenize(text.split())
