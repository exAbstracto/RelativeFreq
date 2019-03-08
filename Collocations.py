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

        # Do we want to apply the general pre-procesing? (convert all to lowercase,
        # remove unicode characters, remove diacritics, remove punctuation, remove digits?)
        flagPreProcess = boolOption('Do you want to pre-process text (convert to lowercase, '
                                    'remove unicode characters, remove diacritics, remove punctuation, remove digits) ? ')
        if flagPreProcess == 1:
            words = preProcess(words)

        # Do we want to apply stemming?
        flagApplyStemming = boolOption('Do you want to apply stemming (remove morphological affixes) on corpus ? ')
        if flagApplyStemming == 1:

            # Apply the Snowball stemmer
            words = doStemming(words)

        # Which method to apply collocations ?
        bigramMethod = -1
        while bigramMethod not in [0,1,2]:
            bigramMethod = int_option('Which method to apply collocations to corpus ? '
                                      '(0=''DICTIONARY'', 1=''REGEX'', 2=''FULL SCAN'' (default 0) ')
        flagProceed = 1
        if bigramMethod in [1, 2]:
            flagProceed = boolOption('This method is VERY slow and it will take a long time on '
                                     'large corpora of text. Are you sure you want to proceed? ')

        if flagProceed == 1:

            # Results
            results = {}

            for i in range(10):

                results[i] = []
                logging.info('FINDING COLLOCATIONS ----> STEP %s' %(i+1))

                # Now let's find collocations
                words = findCollocations(words, bigramMethod)

                # Save the new text, after applying the bi-grams found in this step
                # saveToFile(text=' '.join(words),
                saveToFile(text=words,
                           folderName=corpus.split('.')[0],
                           fileName=corpus.split('.')[0] + '_step_' + str(i + 1),
                           suffix='')

                # collocations = [word for word in words if word.count('_') == i+1]
                collocations = [word for word in words if word.count('_') > 0]

                # Now let's find the relative frequencies
                dictionary = None
                dictionary = buildDictionary(collocations, freqType=1)
                if dictionary:

                    results[i] = [len([word[0] for word in dictionary.most_common() if word[0].count('_') == j+1]) for j in range(10)]
                    results[i].append([len([word[0] for word in dictionary.most_common() if word[0].count('_') > 10 ])])

                    # Let's display the 100 most frequent words
                    showMostFrequent(dictionary, 100, type=1)

                    # Let's save the dictionary to disk
                    # We create a new folder named after the corpus and store the resulting files there
                    saveToCSVFile(text='\n'.join('%s\t%s' % word for word in dictionary.most_common()),
                    # saveToFile(text=[word for word in dictionary.most_common()],
                               folderName=corpus.split('.')[0],
                               fileName=corpus.split('.')[0] + '_collocations_step_' + str(i+1),
                               suffix='')

            logging.info('========== SUMMARY ==========')
            for i in range(10):
                logging.info('Step %s: \t%s' %(i+1, results[i]))