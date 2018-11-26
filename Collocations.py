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
        flagRemoveStopWords = boolOption('Do you want to remove the stopwords from corpus? ')
        if flagRemoveStopWords == 1:

            # Let's ask our user to supply the stopwords file name
            stopWordsFile = None
            stopWordsFile = stringOption('Stopwords file? [stopwords.txt]: ', None, 'stopwords.txt')
            if stopWordsFile:

                # Load the stopwords from the provided file
                stopwords = None
                stopwords = loadWords(stopWordsFile)
                if stopwords:

                    # Remove the stopwords from the original corpus
                    words = removeStopwords(words, stopwords)

        # Do we want to apply the general pre-procesing? (convert all to lowercase, remove unicode characters, remove diacritics, remove punctuation, remove digits?)
        flagPreProcess = boolOption('Do you want to pre-process text (convert to lowercase, remove unicode characters, remove diacritics, remove punctuation, remove digits) ? ')
        if flagPreProcess == 1:
            words = preProcess(words)

        # Do we want to apply stemming?
        flagApplyStemming = boolOption('Do you want to apply stemming (remove morphological affixes) on corpus ? ')
        if flagApplyStemming == 1:

            # Apply the Snowball stemmer
            words = doStemming(words)

        for i in range(10):
            logging.info('FINDING COLLOCATIONS ----> STEP %s' %(i+1))

            # Now let's find collocations
            words = findCollocations(words)

            collocations = [word for word in words if word.count('_') == i+1]
            # Now let's find the relative frequencies
            dictionary = None
            dictionary = buildDictionary(collocations, freqType=1)
            if dictionary:

                # Let's display the 20 most frequent words

                showMostFrequent(dictionary, 100)

                # Let's save the dictionary to disk
                # We create a new folder named after the corpus and store the resulting files there
                saveDictionary(dictionary, corpus.split('.')[0], corpus.split('.')[0] + '_collocations_step_' + str(i+1), '')
