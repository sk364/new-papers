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
from process_data import get_most_similar_docs


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


@app.route("/search", methods=["POST"])
def search():
    req_json = request.json
    abstract = req_json.get("abstract")

    data = []
    if abstract:
        data = get_most_similar_docs(abstract).to_dict('records')[:100]
        categories = [] # get_categories(abstract)

    return json.dumps({
        'data': data,
        'categories': categories
    })
