# New Papers
A recommendation and categorization application to categorize and suggest related research papers, given an abstract.

![screenshot](screenshot.png?raw=true "Screenshot")

## Table of Contents

* [Installation](#installation)
* [How to Run?](#run)
* [Files](#files)
* [Overview & Motivation](#overview)
* [Analysis](#analysis)
* [Conclusion](#conclusion)
* [Acknowledgements](#ack)

## Installation<a name="installation"></a>

Python version required is `>=3.7`. Below are the libraries used:

* pandas
* scikit-learn
* dask
* nltk
* flask

Install all libraries using this command: `pip install -r requirements.txt`.

## How to Run?<a name="run"></a>

* Clone the repository.
* Create directories `data` & `models`, if not already present.
* Download the dataset from [here](https://www.kaggle.com/Cornell-University/arxiv) and move it inside the `data` directory naming it `dataset.json`.
* Run the application: `flask run`
* Go to `http://localhost:5000/setup` to setup the data models.
* Now, navigate to `http://localhost:5000/` and the app is ready to serve!

## Files<a name="files"></a>

The following is the list of files hosted in this repository:

* `app.py`: flask application module containing the routes
* `templates`: directory containing the HTML files
* `etl.py`: data processing module
* `requirements.txt`: project requirements
* `models`: directory containing the processed data and models
* `data`: directory containing the dataset JSON file

## Overview & Motivation<a name="overview"></a>

A research paper is a piece of academic writing containing original research results or an interpretation of existing results. The papers, even just the abstract text, are many a times really long and complex to understand, in the first glance. Basically, it's a time-intensive process. While there are softwares which can auto-summarize articles, it still takes a good amount of effort to go through a vast number of articles to find which are related to the paper in question.

This project attempts to solve the problem of sifting through a myriad number of articles to filter the relevant articles for the study, by exposing a web application in which the user can provide an abstract of an article to get a list of related articles. The application uses the [arXiv research paper dataset](https://www.kaggle.com/Cornell-University/arxiv) to perform content-based filtering to recommend the related articles.

## Analysis<a name="analysis"></a>

### Data Exploration

### Data Visualization

## Conclusion<a name="conclusion"></a>

### Reflection

### Improvement

## Acknowledgements<a name="ack"></a>

Thanks to [Kaggle](https://www.kaggle.com/), [Cornell University](https://www.kaggle.com/Cornell-University), [arXiv](https://arxiv.org) for providing the dataset and [Udacity](https://www.udacity.com/) for delivering the data science course materials.
