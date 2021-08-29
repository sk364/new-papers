import joblib
import json

import pandas as pd
import dask.bag as db
from dask.bag import random

DATASET_FILE_PATH = "data/dataset.json"
DATAFRAME_SAVE_PATH = "models/df.json"

MIN_YEAR = 2019
MAX_PAPERS = 10000


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

    return pd.DataFrame(data.filter(filter_lambda).map(trim_lambda))


def get_dummies(df):
    """Returns list of dummy columns"""
    return df.general_category.str.get_dummies(sep=' ')


def transform_data(df):
    """Transforms data, building general category, one-hot-encoding general
    category and finally dropping the categories column"""

    df['general_category'] = df.categories.apply(
        lambda item: " ".join(
            list(
                set(
                    [category.split(".")[0] for category in item.split(' ')]
                )
            )
        )
    )

    # concat dummy columns for general categories
    dummies = get_dummies(df)
    df = pd.concat([df, dummies], axis=1)

    # drop `categories` column
    df.drop(columns=['categories'], inplace=True)

    return df


def load_data_from_json(file_path):
    """Loads data from JSON file stored at `file_path`"""
    return db.read_text(file_path).map(json.loads)


def save_dataset(df, save_to):
    df.to_json(path_or_buf=save_to, orient='index')


if __name__ == "__main__":
    df = load_data_from_json(DATASET_FILE_PATH)
    df = filter_data(df)
    df = transform_data(df)
    save_dataset(df, DATAFRAME_SAVE_PATH)
