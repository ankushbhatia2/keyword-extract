KeyWord Extractor Python

This is a simple library to extract keywords from a document using a corpus. Built in python 3.5.

__DEPENDENCIES__:
    
    NLTK

__INSTALLATION__:
    
    Installing dependencies:
        pip install --upgrade nltk
        In python Interpreter, run:
            import nltk
            nltk.download()
        Download the required packages : (pos_tagger, stop_words(optional))

    Installing the package:
        Download or clone the package.
        On command line, run : python setup.py install

__USAGE__ :

    from keywords import KeyWords
    from nltk.corpus import stopwords

    with open('script.txt', 'r', encoding="utf8") as f:
        data = f.read()

    with open('transcript_1.txt', 'r', encoding="utf8") as f1:
        corpus_1 = f1.read()

    stopWords = stopwords.words('english')
    keyword = KeyWords(corpus=corpus_1, stop_words=stopWords, alpha=0.8)
    d = keyword.get_keywords(data, n=20)
    for i in d:
        print("Keyword : %s \n Score : %f" %(i[0], i[1]))

Functionalities:

__init__ :
    
    Arguements:
        corpus (optional) : Add your own corpus for prioritizing keywords based on the corpus.
        stop_words : Your own stop_words list
        alpha (default 0.5) : Alpha is the weighting factor which decides how much weightage you want to give to the corpus.
__get_keywords__ :
    
    Arguements :
        text : Your data
        n(default 20) : Number of keywords you want to extract
