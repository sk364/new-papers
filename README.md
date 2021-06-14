# New Papers
A recommendation and categorization application to categorize and suggest related research papers, given an abstract.

!TODO: add screenshots

## Table of Contents

* [Installation](#installation)
* [Files & Data](#files)
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
* matplotlib
* flask

Install all libraries using this command: `pip install -r requirements.txt`.

## Files & Data<a name="files"></a>

The repository hosts a Flask app which structures the project in the following image showing a tree:

!TODO: add image

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
