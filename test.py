# -*- coding: utf-8 -*-
import sys
import codecs
if sys.stdout.encoding != 'cp850':
    sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'cp850':
    sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')


from keywords import KeyWords
from nltk.corpus import stopwords

with open('script.txt', 'r') as f:
    data = f.read()

with open('transcript_1.txt', 'r', encoding="utf8") as f1:
    corpus_1 = f1.read()

with open('transcript_2.txt', 'r', encoding="utf8") as f2:
    corpus_2 = f2.read()

with open('transcript_3.txt', 'r', encoding="utf8") as f3:
    corpus_3 = f3.read()

stopWords = stopwords.words('english')
keyword = KeyWords(corpus=corpus_1, stop_words=stopWords, alpha=0.8)
d = keyword.get_keywords(data, n=20)
for i in d:
    print("Keyword : %s \n Score : %f" %(i[0], i[1]))