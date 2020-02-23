import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    raw_embeddings = []
    for line in open(embeddings_path, encoding='utf-8'):
        raw_embeddings.append(line.strip().split('\t'))

    embeddings = {word_vector[0]:np.array(word_vector[1:]).astype(np.float32) 
                  for word_vector in raw_embeddings}

    embeddings_dim = len(next(iter(raw_embeddings))[1:])

    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """
    Transforms a string to an embedding by averaging word embeddings.
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    question = question.split()
    question_len = len(question)
    
    question_vector = []
    word_not_exist = 0
    for word in question:
        try:
            embedding_val = embeddings[word]
        except KeyError:
            embedding_val = np.zeros(dim)
            word_not_exist += 1
        question_vector.append(embedding_val)
        
    question_mean = np.sum(question_vector, axis=0) / (question_len - word_not_exist)
    if np.isnan(question_mean).any():
        question_mean = np.zeros(dim)
    return question_mean.reshape(1,-1)
    

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
