import json

from flask import Flask, render_template, request
from etl import setup, get_similar_papers


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
        data = get_similar_papers(abstract)

    return json.dumps(data)
