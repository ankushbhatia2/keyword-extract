import re
from nltk import RegexpParser, pos_tag
class KeyWords(object):
    def __init__(self, corpus=None, stop_words=[], alpha=0.5):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha should be between 0-1")
        self.stop_words = stop_words
        stop_word_regex_list = []
        for word in self.stop_words:
            word_regex = r'\b' + word + r'(?![\w-])'
            stop_word_regex_list.append(word_regex)
        self.stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
        self.corpus = corpus
        self.alpha = alpha
        self.parser = RegexpParser('''
                            KEYWORDS: {<DT>? <JJ>* <NN.*>+}
                            P: {<IN>}
                            V: {<V.*>}
                            PHRASES: {<P> <KEYWORDS>}
                            ACTIONS: {<V> <KEYWORDS|PHRASES>*}
                            ''')

    def is_number(self, s):
        try:
            float(s) if '.' in s else int(s)
            return True
        except ValueError:
            return False

    def _sentence_tokenize(self, text):
        sentence_split = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013\n]|\\s\\-\\s')
        sentences = sentence_split.split(text)
        return sentences

    def _phrase_tokenize(self, sentences):
        phrase_list = []
        for s in sentences:
            tmp = re.sub(self.stop_word_pattern, '|', s.strip())
            phrases = tmp.split("|")
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != "":
                    phrase_list.append(phrase)
        phrase_list_new = []
        for p in phrase_list:
            tags = pos_tag(self._word_tokenize(p))
            if tags != []:
                chunks = self.parser.parse(tags)
                for subtree in chunks.subtrees(filter=lambda t: t.label() == 'KEYWORDS'):
                    keyword = ' '.join([i[0] for i in subtree])
                    phrase_list_new.append(keyword)


        return phrase_list_new


    def _word_tokenize(self, text):
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for single_word in splitter.split(text):
            current_word = single_word.strip().lower()
            if current_word != '' and not self.is_number(current_word):
                words.append(current_word)
        return words

    @property
    def _corpus_keywords(self):
        if self.corpus:
            sents = self._sentence_tokenize(self.corpus)
            return self._phrase_tokenize(sents)
        else:
            return None


    def compute_word_scores(self, phraseList):
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = self._word_tokenize(phrase)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
        return word_score


    @property
    def _corpus_keyword_scores(self):
        corp_keywords = self._corpus_keywords
        if corp_keywords:
            word_scores = self.compute_word_scores(corp_keywords)
            keyword_candidates = {}
            for phrase in corp_keywords:
                keyword_candidates.setdefault(phrase, 0)
                word_list = self._word_tokenize(phrase)
                candidate_score = 0
                for word in word_list:
                    candidate_score += word_scores[word]
                keyword_candidates[phrase] = candidate_score
            return keyword_candidates
        else:
            return None

    def phrase_scroing(self, phrase_list, word_score):
        corp_scores = self._corpus_keyword_scores
        keyword_candidates = {}
        for phrase in phrase_list:
            keyword_candidates.setdefault(phrase, 0)
            word_list = self._word_tokenize(phrase)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            if corp_scores:
                keyword_candidates[phrase] = (1-self.alpha)*candidate_score + (self.alpha)*(corp_scores[phrase] if phrase in corp_scores else 0.0)
            else:
                keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def get_keywords(self, text, n=20):
        sentence_list = self._sentence_tokenize(text)
        phrase_list = self._phrase_tokenize(sentence_list)
        word_scores = self.compute_word_scores(phrase_list)
        keyword_candidates = self.phrase_scroing(phrase_list, word_scores)
        return sorted(keyword_candidates.items(), key=lambda x: x[1], reverse=True)[:n]