import logging, sys

# Configure logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# The functions used here are defined in the separate file Functions.py
from Functions import *

# --------------------------------------------------------
# Here we go - this is where the actual execution starts !
# --------------------------------------------------------

# Let's ask our user to supply the corpus file name
corpus = None
corpus = stringOption('Corpus file? [corpus.txt]: ', None, 'corpus.txt')
if corpus:

    # Let's load the individual words in memory
    words = None
    words = loadWords(corpus)
    if words:

        # Do we want to remove stopwords?
        remove_stop = boolOption('Do you want to remove the stopwords from the corpus? ')
        if remove_stop == 1:

            # Let's ask our user to supply the stopwords file name
            stopwords_file = None
            stopwords_file = stringOption('Stopwords file? [stopwords.txt]: ', None, 'stopwords.txt')
            if stopwords_file:

                # Load the stopwords from the provided file
                stopwords = None
                stopwords = loadWords(stopwords_file)
                if stopwords:

                    # Remove the stopwords from the original corpus
                    words = removeStopwords(words, stopwords)

        # Now let's find the relative frequencies
        dictionary = None
        dictionary = buildDictionary(words)
        if dictionary:

            # Let's display the 20 most frequent words
            showMostFrequent(dictionary, 20)

            # Let's save the dictionary to disk
            # We create a new folder named after the corpus and store the resulting files there
            saveDictionary(dictionary, corpus.split('.')[0], corpus.split('.')[0], '')
