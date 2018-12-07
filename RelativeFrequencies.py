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
    words = []
    words = loadWords(corpus)
    words_aux = [word for word in words if len(word)>1]
    words.clear()
    words = words_aux.copy()
    words_aux.clear()

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


        # Do we want to apply the general pre-procesing? (convert all to lowercase, remove unicode characters, remove diacritics, remove punctuation, remove digits?)
        flagPreProcess = boolOption('Do you want to pre-process text (convert to lowercase, remove unicode characters, remove diacritics, remove punctuation, remove digits) ? ')
        if flagPreProcess == 1:
            words = preProcess(words)

        # Do we want to apply stemming?
        flagApplyStemming = boolOption('Do you want to apply stemming (remove morphological affixes) on corpus ? ')
        if flagApplyStemming == 1:

            # Apply the Snowball stemmer
            words = doStemming(words)


        # Now let's find the relative frequencies
        dictionary = None
        dictionary = buildDictionary(words)
        if dictionary:

            # Let's display the 20 most frequent words
            showMostFrequent(dictionary, 20)

            # Let's save the dictionary to disk
            # We create a new folder named after the corpus and store the resulting files there
            # for word in dictionary.most_common():
            #     print('{} {:0.10f}'.format(word[0], word[1]))
            saveToFile(text='\n'.join('%s\t%.10f' % (word[0], word[1]) for word in dictionary.most_common()),
                      type=0,
                      folderName=corpus.split('.')[0],
                      fileName=corpus.split('.')[0],
                      suffix='')
