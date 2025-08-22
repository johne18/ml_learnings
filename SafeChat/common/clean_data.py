import emoji
import nltk
from nltk.corpus import stopwords


def contains_emoji(text):
    return bool(emoji.emoji_count(text))


def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')


def get_stop_words():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['I', 'You', 'YOU', 'like', 'The'])
    return stop_words