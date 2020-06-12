import re
import string
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

class Models(object):
    """docstring for Models."""

    def __init__(self):
        super(Models, self).__init__()
        self._s = stop_words.ENGLISH_STOP_WORDS
        self._p = frozenset(string.punctuation)
        self._l = WordNetLemmatizer()
        self._v = pickle.load(open('./models/vector.pkl', 'rb'), encoding="latin1")
        self._c = CountVectorizer(vocabulary = pickle.load(open('./models/count_vector.pkl', 'rb')))
        self._t = pickle.load(open('./models/tfidf.pkl', 'rb'))
        self._knn = pickle.load(open('./models/knn_model.pkl', 'rb'), encoding = 'latin1')
        self._svm = pickle.load(open('./models/svm_model.pkl', 'rb'))
        self._nb = pickle.load(open('./models/nb_model.pkl', 'rb'))
        self._cnn = pickle.load(open('./models/softmax_model.pkl', 'rb'))

    def map_name(self, i, is_knn = True):
        if(is_knn):
            m = {
                1 : 'World',
                2 : 'Sports',
                3 : 'Business',
                4 : 'Sci/Tech'
            }
        else:
            m = {
                0 : 'Sports',
                1 : 'World',
                3 : 'Business',
                6 : 'Sci/Tech'
            }
        return m[i] if i in m else m[1]

    def classify_knn(self, text):
        tmp1 = ' '.join( word for word in text.strip().lower().split() if word not in self._s )
        tmp2 = ''.join( ch for ch in tmp1 if ch not in self._p )
        clean = ' '.join(re.sub('\\d+', '', ' '.join( self._l.lemmatize(word) for word in tmp2.split() )).split())
        v = self._v.transform([clean])
        return self.map_name(self._knn.predict(v)[0])

    def classify_svm(self, text):
        tmp1 = self._c.transform([text])
        tmp2 = self._t.transform(tmp1)
        return self.map_name(self._svm.predict(tmp2)[0], False)

    def classify_nb(self, text):
        tmp1 = self._c.transform([text])
        tmp2 = self._t.transform(tmp1)
        return self.map_name(self._nb.predict(tmp2)[0], False)

    def classify_cnn(self, text):
        tmp1 = self._c.transform([text])
        tmp2 = self._t.transform(tmp1)
        return self.map_name(self._cnn.predict(tmp2)[0], False)

    def predict(self, text):
        return [
            self.classify_svm(text),
            self.classify_nb(text),
            self.classify_knn(text),
            self.classify_cnn(text)
        ]
