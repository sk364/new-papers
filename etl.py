import sys
import joblib
import json
import dask.bag as db
import pandas as pd
import nltk
import re
import time

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dask.bag import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATASET_FILE_PATH = "data/dataset.json"
MATRIX_SAVE_PATH = "models/matrix.pkl"
VECTORIZER_SAVE_PATH = "models/vectorizer.pkl"
DATAFRAME_SAVE_PATH = "models/df.json"
MIN_YEAR = 2019
MAX_PAPERS = 100000
NUM_TOP_ITEMS = 101


def tokenize(text):
    """Tokenizes the given `text`"""

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens if word not in stop_words
    ]

    return tokens


def load_data_from_json(file_path):
    """Loads data from JSON file stored at `file_path`"""

    return db.read_text(file_path).map(json.loads)


def get_latest_version(row):
    """Extracts latest version, parsing the year from it.
    If no latest version found, then default 1900 year is returned"""

    created = "Mon, 1 Jan 1900 00:00:00 GMT"
    journal_ref = row.get('versions', [])
    if len(journal_ref) > 0:
        latest_version = journal_ref[-1]
        created = latest_version.get('created', "Mon, 1 Jan 1900 00:00:00 GMT")

    try:
        return int(created.split(' ')[3])
    except IndexError:
        return 1900

    return 1900


def filter_data(data):
    """Filters data, selecting items created after the `MIN_YEAR`,
    returning back a pandas DataFrame object"""

    filter_lambda = lambda item: get_latest_version(item) > MIN_YEAR
    trim_lambda = lambda item: {
        'id': item['id'],
        'title': item['title'],
        'categories': item['categories'],
        'abstract': item['abstract']
    }

    sample_size = MAX_PAPERS
    while True:
        try:
            return pd.DataFrame(
                random.sample(
                    data.filter(filter_lambda).map(trim_lambda), sample_size
                ).compute()
            )
        except:
            sample_size /= 10


def clean_data(df):
    """Cleans data, extracting the general category for
    each category of an item"""

    df['general_category'] = df.categories.apply(
        lambda item: " ".join(
            list(
                set(
                    [category.split(".")[0] for category in item.split(' ')]
                )
            )
        )
    )

    return df


def compute_similarities(X, Y):
    """Computes similarities using pairwise linear kernel method
    between X and Y matrices"""

    return linear_kernel(X, Y)


def save_matrix(matrix):
    """Saves TF-IDF matrix using joblib"""
    joblib.dump(matrix, MATRIX_SAVE_PATH)


def save_vectorizer(vectorizer):
    """Saves TF-IDF matrix using joblib"""
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)


def save_dataset(df):
    """Saves the DataFrame as a json file"""
    df.to_json(path_or_buf=DATAFRAME_SAVE_PATH, orient='index')


def preprocess(df):
    """Compute TF-IDF matrix and save it and the `df`"""

    tfidf_vect = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = tfidf_vect.fit_transform(df['abstract'])
    save_dataset(df)
    save_vectorizer(tfidf_vect)
    save_matrix(tfidf_matrix)


def get_similar_papers(abstract):
    """Returns list of similar papers for the given `abstract`"""

    # load the saved dataset and the TF-IDF matrix
    df = pd.read_json(DATAFRAME_SAVE_PATH, orient="index")
    saved_tfidf_vectorizer = joblib.load(VECTORIZER_SAVE_PATH)
    saved_tfidf_matrix = joblib.load(MATRIX_SAVE_PATH)

    # rebuild TF-IDF vectorizer using saved matrix's vocabulary
    # and fit-transform the new abstract text over it
    tf_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        vocabulary=saved_tfidf_vectorizer.vocabulary_
    )
    tfidf_matrix = tf_vectorizer.fit_transform([abstract])

    # compute similarities matrix and get top N indices from it
    similarities = compute_similarities(tfidf_matrix, saved_tfidf_matrix)
    top_N_indices = similarities[0].argsort()[:-NUM_TOP_ITEMS:-1]

    # build the results object with items' metadata
    similar_items = []
    for index in top_N_indices:
        similar_items.append(dict(df.iloc[index, :]))

    return similar_items


def setup():
    """Load, preprocess and save the formatted data"""
    start_time = time.time()

    nltk.download([
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ])

    print("Loading data...")
    documents = load_data_from_json(DATASET_FILE_PATH)

    df = filter_data(documents)
    df = clean_data(df)

    print("Saving processed data...")
    preprocess(df)

    print("Successfully saved data in `models` directory")
    print(f"Took {time.time() - start_time} seconds")


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        print('Usage: python etl.py setup')
        sys.exit(-1)

    if args[1] == "setup":
        setup()
    else:
        print('Usage: python etl.py setup')
