import json
import pandas as pd
import time

from flask import Flask, render_template, request
from etl import (
    setup,
    get_similar_papers,
    get_categories,
    DATAFRAME_SAVE_PATH,
    TEST_SAMPLE_SIZE
)


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/setup')
def app_setup():
    try:
        setup()
        return "Success"
    except:
        return "Something went wrong!"


@app.route("/report")
def report():
    # load the saved dataset
    df = pd.read_json(DATAFRAME_SAVE_PATH, orient="index")

    # take a sample from the processed dataset
    df_test = df.sample(TEST_SAMPLE_SIZE, replace=False)

    start_time = time.time()
    num_no_related = 0
    num_related_papers = 0

    for _, row in df_test.iterrows():
        related_papers = get_similar_papers(row['abstract'])

        # a paper will always find itself as similar to it, so remove it
        related_papers = [
            paper for paper in related_papers if paper['id'] != row['id']
        ]

        # if no papers found, increase the counter,
        # otherwise add number of papers found to the count
        if len(related_papers) == 0:
            num_no_related += 1
        else:
            num_related_papers += len(related_papers)

    # compute metrics
    total_time = time.time() - start_time
    avg_time = total_time / TEST_SAMPLE_SIZE
    avg_num_related_papers = num_related_papers / TEST_SAMPLE_SIZE
    percent_num_no_related = (num_no_related / TEST_SAMPLE_SIZE) * 100

    return render_template(
        "report.html",
        num_sample_items=TEST_SAMPLE_SIZE,
        avg_time=avg_time,
        avg_num_related_papers=avg_num_related_papers,
        percent_num_no_related=percent_num_no_related
    )


@app.route("/search", methods=["POST"])
def search():
    req_json = request.json
    abstract = req_json.get("abstract")

    data = []
    if abstract:
        data = get_similar_papers(abstract)
        categories = get_categories(abstract)

    return json.dumps({
        'data': data,
        'categories': categories
    })
