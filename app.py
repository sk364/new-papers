import json

from flask import Flask, render_template, request
from process_data import get_most_similar_docs


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


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
