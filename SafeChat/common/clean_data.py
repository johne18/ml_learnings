import emoji

import nltk
from nltk.corpus import stopwords, words
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def contains_emoji(text):
    return bool(emoji.emoji_count(text))


def filter_non_standard_words(text):
    nltk.download('words', quiet=True)

    english_words = set(words.words())
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.lower() in english_words]

    return ' '.join(filtered)


def generate_grams(text, n):
    nltk.download('punkt_tab', quiet=True)

    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))

    return n_grams


def get_stop_words():
    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    stop_words.update(['I', 'You', 'YOU', 'like', 'The'])

    return stop_words


def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')