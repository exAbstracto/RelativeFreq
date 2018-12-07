# First, import the python libraries we're going to use
import logging, os, collections, sys, re
import regex
# import gensim
from nltk import tokenize, collocations, stem
from langdetect import detect

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# Constants
ctPunctuationTokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}',
                          '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
ctPunctuation = '?.!/;:()&+%'
ctDigits = '0123456789'
ctLanguages = {
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'hu': 'hungarian',
        'it': 'italian',
        'ro': 'romanian'
    }

# Configure logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# --------------------------------------------------------------------------------------------------
# A function to load all sentences from a text file (any text file)
# We'll use it for loading in memory all sentences from the supplied corpus
# --------------------------------------------------------------------------------------------------
def loadSentences(fileName):
    """ Split corpus in sentences
    :param fileName: File containing corpus body
    :return:
    """
    originalSentences = list()
    if fileName:
        logging.info("Loading corpus sentences...")
        try:
            originalSentences = tokenize.sent_tokenize(
                text=open(fileName, mode='r', encoding='utf-8').read().lower(), language='english')
            logging.info("%s sentences loaded from file %s [%0.3f Mb].",
                         '{:,}'.format(len(originalSentences)), fileName,
                         os.path.getsize(fileName) / (1024 * 1024))
        except Exception as e:
            logging.info(repr(e))
    else:
        logging.info("Please provide a corpus file.")
    return originalSentences


# # -------------------------------------------------------------------
# # A function to split corpus sentences in words
# # -------------------------------------------------------------------
# def sentencesToWords(sentences):
#     for sentence in sentences:
#         # yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
#         yield (gensim.utils.simple_preprocess(str(sentence), min_len=1, max_len=100, deacc=False))  # deacc=True removes punctuations


# --------------------------------------------------------------------------------------------------
# A function to load all individual words from a text file (any text file)
# We'll use it for loading in memory all words from the supplied corpus or from the stopwords file
# --------------------------------------------------------------------------------------------------
def loadWords(fileName):
    """
    :param fileName: Corpus of text, as txt file
    :return: iterable of words
    """
    words = []
    if fileName and os.path.exists(fileName):
        logging.info("Loading words from file %s [%0.3f Mb].", fileName, os.path.getsize(fileName) / (1024 * 1024))
        try:
            # # words = tokenize.word_tokenize(text=open(fileName, mode='r', encoding='utf-8').read(), language='english')
            words  =  re.findall(r'\w+', open(fileName, mode='r', encoding='utf-8').read().lower())
            logging.info("%s words loaded...", '{:,}'.format(len(words)))
        except Exception as e:
            # logging.info("Please provide a valid file name.")
            logging.info(repr(e))
    else:
        logging.info("Please provide a valid file name.")

    # Remove potential void words
    words = [word for word in words if word !='']

    return words


# --------------------------------------------------
# A function to remove a set of words from a corpus
# We'll use it to remove the stopwords
# --------------------------------------------------
def removeStopwords(words, stopwords):
    """ Remove stopwords from corpus
    :return: list of words minus stopwords
    """
    wordsAux = []
    if stopwords and words:
        logging.info("Removing stopwords...")
        wordsAux = [x for x in words if x not in stopwords]
        logging.info("%s words retained from text.", '{:,}'.format(len(wordsAux)))
    return wordsAux


# --------------------------------------------------
# A function to remove unicode punctuation
# --------------------------------------------------
def removeUnicodePunctuation(text):
    """ Removes unicode punctuation
    :param text:
    :return: string without unicode punctuation
    """
    res = text
    res = res.replace(u'”', ' ')
    res = res.replace(u'’', ' ')
    res = res.replace(u'…', ' ')
    res = res.replace(u'„', ' ')
    res = res.replace(u'“', ' ')
    res = res.replace(u',', ' ')
    return res


# --------------------------------------------------
# A function to remove Romanian diacritics
# --------------------------------------------------
def removeDiacritics(text):
    """ Replaces diacritics
    :param text:
    :return: string without diacritics
    """
    res = text
    res = res.replace(u'ț', 't')
    res = res.replace(u'ă', 'a')
    res = res.replace(u'î', 'i')
    res = res.replace(u'ș', 's')
    res = res.replace(u'â', 'a')
    res = res.replace(u'ţ', 't')
    res = res.replace(u'ş', 's')
    res = res.replace(u'à', 'a')
    return res

# -----------------------------------------------------------------
# A function to do some generic pre-processing on a corpus of words
# -----------------------------------------------------------------
def preProcess(document, lowercase=True, unicode=True, diacritics=True, punctuation=True, digits=True):
    """ Pre-process corpora for training
    :param sentences: The corpora sentences (list)
    :param lowercase: change all text to lowercase? (True/False, default = True)
    :param unicode: remove Unicode punctuation? (True/False, default = True)
    :param diacritics: remove diacritics? (True/False, default = True)
    :param punctuation: remove punctuation? (True/False, default = True)
    :param digits: remove digits? (True/False, default = True)
    :return: preprocessed list of words
    """

    numWords = len(document)
    i = 1

    logging.info('Pre-processsing %s words...', '{:,}'.format(numWords))

    # replace uppercase with lowercase
    if lowercase:
        document = [word.lower() for word in document]

    # remove unicode punctuation
    if unicode:
        document = [removeUnicodePunctuation(word) for word in document]


    # remove diacritics
    if diacritics:
        document = [removeDiacritics(word) for word in document]

    # filter punctuation, stopwords, digits
    if punctuation:
        document = [word for word in document if word not in ctPunctuationTokens]
        document = [re.sub('[' + ctPunctuation + ']', '', word) for word in document]

    # remove digits
    if digits:
        document = [re.sub('[' + ctDigits + ']', '', word) for word in document]

    logging.info('Pre-processing of %s words finished successfully!', '{:,}'.format(numWords))
    logging.info('%s words remaining', '{:,}'.format(len(document)))

    # Remove potential void words
    document = [word for word in document if word.strip() != '']

    return document


# ------------------------------------------------------------------------
# A function to remove morphological affixes from corpus (Snowball stemmer)
# ------------------------------------------------------------------------
def doStemming(words):
    stemmed = []
    if words:
        detectedLanguage = detect(' '.join(words))
        if detectedLanguage in ctLanguages:
            stemmerLanguage = ctLanguages[detectedLanguage]
        else:
            stemmerLanguage = 'english'
        stemmer = stem.SnowballStemmer(stemmerLanguage)
        for word in words:
            stemmed.append(stemmer.stem(word))
    return stemmed


# ------------------------------------------------------------------------
# A function to find and mark collocations (bi-grams) in a corpus of text
# ------------------------------------------------------------------------
# @profile
def findCollocations(words, bigramMethod):
    if words:
        # Find collocations (Manning's algorithm, NLTK)
        bigramMeasures = collocations.BigramAssocMeasures()
        finder = collocations.BigramCollocationFinder.from_words(words=words, window_size=2)
        finder.apply_freq_filter(2)
        topBigrams = finder.nbest(bigramMeasures.pmi, 10000000)

        # Mark collocations in corpus
        if bigramMethod == 0:

            # ---------------------------------------------------------------------------------------
            # METHOD 3 - DICTIONARY
            # Put bi-grams in a dictionary then loop over the whole corpus once and
            # check if two successive words are part of a bi-gram from the dictionary
            # ---------------------------------------------------------------------------------------
            # +++ Very fast !
            # ---
            # ---------------------------------------------------------------------------------------
            topBigramsDict = {}
            for bigram in topBigrams:
                if not bigram[0] in topBigramsDict:
                    topBigramsDict[bigram[0]] = [bigram[1]]
                else:
                    topBigramsDict[bigram[0]].append(bigram[1])

            # Now mark collocations in corpus
            document = []
            skipIndex = -1
            for index, word in enumerate(words):
                if index != skipIndex and \
                        index < len(words) - 1 and \
                        word and words[index + 1] and \
                        word != words[index + 1] and \
                        len(word) > 1 and len(words[index + 1]) > 1 and \
                        word not in ctPunctuationTokens and words[index + 1] not in ctPunctuationTokens:

                    if word not in topBigramsDict:
                        document.append(word)
                    elif words[index + 1] not in topBigramsDict[word]:
                        document.append(word)
                    else:
                        document.append(word + '_' + words[index + 1])
                        skipIndex = index + 1

            words.clear()
            words = document

        elif bigramMethod == 1:

            # ---------------------------------------------------------------------------------------------
            # METHOD 1 - REGEX
            # We use a regex to identify & replace all occurrences of the first bi-gram in the full corpus.
            # Then we move to the second bi-gram and so forth...
            # ---------------------------------------------------------------------------------------------
            # +++ We try to use a regex to identify all occurrences of a bi-gram in corpus.
            #     Then only we move to the next bi-gram.
            #     The question is: which bi-gram should we apply first?
            #     By default, we start with the most frequent ones but those are also the most trivial
            # --- Very slow on larger corpora
            #     The sequence of identification of all occurrences is not guaranteed
            # ---------------------------------------------------------------------------------------------
            corpus = ' '.join(words)
            for b1, b2 in topBigrams:
                if b1 and b2 and \
                        b1 != b2 and \
                        len(b1) > 1 and \
                        len(b2) > 1 and \
                        b1 not in ctPunctuationTokens and \
                        b2 not in ctPunctuationTokens:
                    bigramRegex = regex.compile(r'\b%s\b\s{1}\b%s\b' % (b1,b2))
                    corpus = bigramRegex.sub(b1+'_'+b2, corpus)

            words.clear()
            words = tokenize.word_tokenize(text=corpus, language='english')

        elif bigramMethod == 2:
            # ----------------------------------------------------------------------------------------
            # METHOD 3 - FULL SCAN
            # Apply the first bi-gram to the entire corpus, then the second and so forth...
            # ----------------------------------------------------------------------------------------
            # +++ It guarantees that a bi-gram is applied to ALL its occurrences in the text.
            #     Then only we move to the next bi-gram.
            #     The order of applying bi-grams to corpus is guaranteed.
            #     The question is: which bi-gram should be applied first?
            #     By default, we start with the most frequent ones but those are also the most trivial
            # --- Very very slow, even on smaller corpora!
            # ---------------------------------------------------------------------------------------
            for b1, b2 in topBigrams:
                if b1 and b2 and \
                    b1 != b2 and \
                    len(b1) > 1 and \
                    len(b2) > 1 and \
                    b1 not in ctPunctuationTokens and \
                    b2 not in ctPunctuationTokens:

                    document = []
                    skipIndex = -1
                    for index, word in enumerate(words):
                        if index != skipIndex and index < len(words)-1:
                            if word != b1:
                                document.append(word)
                            elif words[index+1] != b2:
                                document.append(word)
                            else:
                                document.append(word+'_'+words[index+1])
                                skipIndex = index+1

                    words.clear()
                    words = document

    return words


# # -------------------------------------------------------------------
# # A function to find and mark bi-grams in a corpus of text
# # -------------------------------------------------------------------
# def findBigrams(sentences):
#     if sentences:
#         texts = sentences
#         data_words = list(sentencesToWords(texts))
#         bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1)  # higher threshold fewer phrases.
#         bigram_model = gensim.models.phrases.Phraser(bigram)
#         sentences_aux = []
#         sentences_aux = [bigram_model[doc] for doc in texts]
#         sentences = []
#         for sentence in sentences_aux:
#             sentences.append(''.join(sentence))
#     return sentences


# ------------------------------------------------------------------------------------------------------------------
# A function to count the relative frequencies of elements in an iterable (an unordered set of generic elements)
# We'll use it to count the absolute frequencies of words in our corpus
# ------------------------------------------------------------------------------------------------------------------
def buildDictionary(setOfWords, freqType=0):
    """ Build dictionary of unique setOfWords, with absolute / relative frequencies
    :param setOfWords: the corpus
    :param freqType: 0 = Relative, 1 = Absolute
    :return:

    :return: collection of setOfWords and frequencies
    """
    dictionary = None
    if setOfWords:
        try:
            dictionary = collections.Counter()
            # Use an auxiliary Counter to compute relative (instead of absolute) frequencies
            dictionary_aux = collections.Counter(setOfWords)
            for key, value in dictionary_aux.items():
                if value > 1:
                    if freqType == 0:
                        dictionary[key] = value / sum(dictionary_aux.values())
                    else:
                        dictionary[key] = value
            dictionary_aux.clear()
            logging.info("Dictionary built. %s words retained.", '{:,}'.format(len(dictionary)))
        except Exception as e:
            logging.info(repr(e))
    return dictionary


# -------------------------------------------------------------------
# A function to save a dictionary (key, value) to a file on disk
# -------------------------------------------------------------------
def showMostFrequent(dictionary, n):
    """ Print most frequent n words from dictionary
    :param n: Number of most frequent n words from dictionary
    """
    if dictionary:
        logging.info('Let''s display the first %s most frequent words, along with their frequencies', n)
        for word in enumerate(dictionary.most_common(n)):
            logging.info("\t%s. %s (%.10f)", '{:,}'.format(word[0] + 1), word[1][0], word[1][1])


# -------------------------------------------------------------------
# A function to save a dictionary (key, value) to a file on disk
# -------------------------------------------------------------------
def saveToFile(text, type, folderName, fileName, suffix):
    """
    :param text: text to be saved to file
    :param fileName: The sub-folder in which we'll save the file
    :param suffix: an optional suffix for the resulting dictionary file name
    """
    if type == 0:
        fileType = 'dictionary'
        extension = '.csv'
    elif type == 1:
        fileType = 'corpus'
        extension = '.txt'
    if text:
        fpath = os.path.join(folderName)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        try:
            fpath = os.path.join(folderName, fileName + suffix + extension)
            logging.info("Saving %s to file " % fileType + fpath)
            with open(fpath, mode='w', encoding='utf-8') as f:
                f.write(text)
                f.close()
        except Exception as e:
            logging.info(repr(e))


# ---------------------------------------------------------------------
# A function to ask the user a question and wait for the user reply
# We'll use it to ask the user to supply the corpus filename
# ---------------------------------------------------------------------
def stringOption(question, options, default):
    """
    :param question: The question you want to ask the user
    :param options: The possible reply options (if any)
    :param default: The default answer (if any)
    :return: The user answer, or the default option
    """
    while True:
        answer = input(question)
        if not answer:
            return default
        else:
            if options:
                if answer in options:
                    return answer
            elif not options:
                return answer

# ---------------------------------------------------------------------
# A function to ask the user a question and wait for a Yes/No reply
# ---------------------------------------------------------------------
def boolOption(question):
    while True:
        answer = input(question + '[Y/n] : ')
        if not answer:
            return 1
        elif answer.lower() == 'y':
            return 1
        elif answer.lower() == 'n':
            return 0


# ---------------------------------------------------------------------
# A function to ask the user a question and wait for a numeric reply
# ---------------------------------------------------------------------
def int_option(question, default=0):
    while True:
        answer = input(question)
        if not answer:
            return default
        else:
            try:
                answer_int = int(answer)
                return answer_int
            except:
                pass