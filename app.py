import json

from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/search", methods=["POST"])
def search():
  data = [{
    "id": 1,
    "title": "title",
    "abstract": "abs"
  },
  {
    "id": 1,
    "title": "title",
    "abstract": "abs"
  }]
  return json.dumps(data)
