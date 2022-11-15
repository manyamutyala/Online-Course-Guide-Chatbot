import string
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('ignore')

nltk.download('punkt')

nltk.download('wordnet')

f = open("train.txt", 'r', errors="Ignore")
raw = f.read()

raw = raw.lower()


#coverts the raw input into list of sentences.
sent_tokens = nltk.sent_tokenize(raw)

#converts the raw input into list of words
word_tokens = nltk.word_tokenize(raw)

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


def get_response(user_response):
    robo_response = ''
    user_response = user_response.lower()
    sent_tokens.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfvec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response
