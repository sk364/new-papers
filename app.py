import joblib
import json

from flask import Flask, render_template, request
from process_data import (
    read_data,
    init_model,
    get_most_similar_docs,
    SAVED_DF_PATH,
    SIMILARITY_MATRIX_SAVE_PATH
)


app = Flask(__name__)
model = init_model()
embeddings = joblib.load(SIMILARITY_MATRIX_SAVE_PATH)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/search", methods=["POST"])
def search():
    req_json = request.json
    abstract = req_json.get("abstract")

    data = []
    if abstract:
        df = read_data(SAVED_DF_PATH)
        data = get_most_similar_docs(df, model, embeddings, abstract, top_k=300)
        categories = [] # get_categories(abstract)

    return json.dumps({
        'data': data,
        'categories': categories
    })
