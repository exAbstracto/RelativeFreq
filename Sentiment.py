import logging, sys, re

# Configure logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# The functions used here are defined in the separate file Functions.py
from Functions import *

# --------------------------------------------------------
# Here we go - this is where the actual execution starts !
# --------------------------------------------------------

# Let's ask our user to supply the corpus file name
corpusFile = None
corpusFile = stringOption('Corpus file? [corpus.txt]: ', None, 'corpus.txt')
if corpusFile:

    lexiconFile = None
    lexiconFile = stringOption('Lexicon file? [lexicon.txt]: ', None, 'lexicon.csv')

    if lexiconFile:

        # Let's load the lexicon
        lexicon = loadLexicon(lexiconFile)

        if lexicon:

            # Let's load the corpus
            corpus = loadCorpus(corpusFile)

            if corpus:
                # Word count
                tokens = len([word for word in re.findall(r'\w+', corpus) if len(word) > 1])
                logging.info('%s words in corpus', '{:,}'.format(tokens))

                outFreq = {}
                sentimentIndex = 0

                for term in lexicon:
                    if term:
                        term_aux = term.replace('?', '.')
                        term_aux = term_aux.replace('*', r'\S+')
                        term_aux = term_aux.replace('|', '')
                        term_aux = r'\b' + term_aux + r'\b'
                        termRegex = re.compile(r'%s' % term_aux, re.MULTILINE | re.IGNORECASE)
                        subjValue = lexicon[term]
                        findings = [x for x in re.findall(termRegex, corpus) if x not in ctExcluded]
                        freq = len(findings)
                        outFreq[term] = [subjValue, freq, tokens, freq / tokens * float(subjValue), findings]
                        sentimentIndex += outFreq[term][3]

                if outFreq:
                    # Let's display the findings
                    logging.info('---------------------------------------------------')
                    logging.info('Corpus %s', corpusFile.split('.')[0])
                    logging.info('Sentiment Index = {:>20.15f}'.format(sentimentIndex))
                    logging.info('---------------------------------------------------')
                    logging.info('{:20} {:>3} {:>10} {:>10} {:>20} {}'.
                                 format('Term', 'Val', 'Abs.freq.', 'Tokens', 'Contribution', 'Occurrences'))
                    for item in lexicon:
                        # logging.info(format, '{:,}'.format(item), lexicon[item], outFreq[item])
                        logging.info('{:20} {:>3} {:>10} {:>10.0f} {:>20.15f} {}'.
                                     format(item, outFreq[item][0], outFreq[item][1], outFreq[item][2],
                                            outFreq[item][3], outFreq[item][4]))

                    # Let's save the dictionary to disk
                    # We create a new folder named after the corpus and store the resulting files there
                    export = ''
                    export += '---------------------------------------------------' + '\n'
                    export += 'Corpus: ' + corpusFile.split('.')[0] + '\n'
                    export += 'Sentiment Index = {:>20.15f}'.format(sentimentIndex) + '\n'
                    export += '---------------------------------------------------' + '\n'
                    export += '\n'
                    export += 'Term;Sentiment value;Absolute frequency;Corpus size (tokens);' \
                              'Contribution of term;Occurrences' + '\n'
                    for item in lexicon:
                        export += '{};{};{};{:10.0f};{:20.15f};{}'.format(item,
                                                                          outFreq[item][0],
                                                                          outFreq[item][1],
                                                                          outFreq[item][2],
                                                                          outFreq[item][3],
                                                                          outFreq[item][4]) + '\n'
                    if export:
                        saveToCSVFile(
                            text=export,
                            folderName=corpusFile.split('.')[0],
                            fileName=corpusFile.split('.')[0],
                            suffix='_sentiment')
