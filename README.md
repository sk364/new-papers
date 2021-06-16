# New Papers
A recommendation application to suggest related research papers, given an abstract.

![screenshot](./assets/screenshot.png?raw=true "Screenshot")

## Table of Contents

* [Installation](#installation)
* [How to Run?](#run)
* [Files](#files)
* [Overview & Motivation](#overview)
* [Problem Statement](#statement)
* [Metrics](#metrics)
* [Analysis](#analysis)
* [Methodology](#meth)
* [Results](#res)
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

## Problem Statement<a name="statement"></a>

The goal of the project is to create a web application to help users find similar articles, given a piece of text. The application is expected to be useful for conducting literature reviews, finding correlation between past researches and the user's own research, etc.

## Metrics<a name="metrics"></a>

The project evaluates the recommendations based on the criteria of score being greater than the set threshold.
For this project, setting the threshold to 0.50 for a score ranging between 0 and 1 (both inclusive). Since all documents are compared with a single document, zero score documents will be filtered out and will not be considered as retrieved documents.

To evaluate the application, a custom metric "quality" is used as there are no pre-existing ratings to judge the results using train-test approach in this knowledge-based recommendation engine.
Using the above threshold value, `quality` of the search results is defined as the fraction of documents above threshold over the total number of documents.

`Quality` = # of documents above threshold / total # of documents

Another useful metric to evaluate the performance is the search time to find the similar articles.

`Search time` = the number of seconds it takes to find and generate a list of articles (excluding the web processing delays).

## Analysis<a name="analysis"></a>

### Data Exploration

The arXiv dataset contains a repository of ~1.9 million articles, with relevant features such as article titles, authors, categories, abstracts, full text PDFs, and more. The analysis limits itself to 100,000 papers published after the year 2019.

Each item has the following keys:

* id: ArXiv ID
* submitter: Who submitted the paper
* authors: Authors of the paper
* title: Title of the paper
* comments: Additional info, such as number of pages and figures
* journal-ref: Information about the journal the paper was published in
* doi: [Digital Object Identifier](https://www.doi.org)
* abstract: The abstract of the paper
* categories: Categories / tags in the ArXiv system
* versions: A version history

The articles are tagged with 155 unique categories and 20 unique 'general categories' (here, the general category is the category extracted before the first dot separator). The general category is not included in the data, hence it will be generated during preprocessing.

### Data Visualization

![screenshot](./assets/items.png?raw=true "Fig. 1")

**Fig. 1** A sample list of items after the extraction and transformation phase

![screenshot](./assets/detail.png?raw=true "Fig. 2")

**Fig. 2** An item in detail view

It can be seen in Fig. 2 that the abstract contains characters relevant in displaying the mathematical formulas using MathJax script, but for the purposes of this analysis, it shall be removed as part of the tokenization process.

![screenshot](./assets/submission.png?raw=true "Fig. 3")

**Fig. 3** A plot showing the number of submissions by submission date

The above figure indicates that the submissions are skewed towards the recent dates.

![screenshot](./assets/category.png?raw=true "Fig. 4")

**Fig. 4** A plot showing the percentage of papers by general category

Most number of papers are tagged with Computer Science and Mathematics by a really great margin, followed by Condensed Matter and Physics, as shown in Fig. 4.

## Methodology<a name="meth"></a>

### Implementation & Techniques Used

During the setup phase of the application, the dataset is transformed using the TF-IDF vectorization technique, removing the stopwords and tokenizing the texts, to generate the word vector features along with their scores. The vector model and the corresponding matrix generated using the dataset is then saved for future use.

Given a paper abstract, a new piece of text provided by the user through an HTML form, the program then loads the saved vector model and the matrix to fit the new piece of text with it. This generates another TF-IDF matrix for the words in the text.
Using cosine similarity on the matrices, the saved and the newly generated one, top N indices are computed. The indices are then used to find the metadata of the paper which are then sent back as a JSON response to be displayed as an HTML list.

## Results<a name="res"></a>

### Model Evaluation and Validation

Using a validation script taking a sample size of 200 and threshold score of 0.40, the following results are observed:

* ~4 'quality' articles are found for each item in the test set.
* it takes on an average ~2 seconds to find the articles for one abstract.
* ~60% articles in the sample have no related articles.

This report can be generated using the web application itself by navigating to the route: `/report`.

### Justification

It can be seen that the application is useful to find good quality similar articles of their choice in a clean list view in a website mode. But, it is limited to arXiv papers with a smaller dataset size. It also has limitations to read in the mathJax format to fully comprehend the ignorance of important mathematical texts.

All in all, the project brings out its usefulness in a limited domain, but a better infrastructure and upgradation of techniques along with a better webview shall provide a really good enhancement to find interesting papers.

## Conclusion<a name="conclusion"></a>

### Reflection

Steps followed to generate the similar articles:

* Load dataset
* Filter dataset to include the articles published after 2019 (at most 100,000)
* Compute the general category of each article
* Compute TF-IDF matrix
* Save model and matrix
* When new abstract is provided, the saved model and matrix is loaded
* Using the saved vocabulary, a new model is created
* The new model generates a TF-IDF matrix for the given abstract
* Cosine similarity technique generates the similarity score between each article and the abstract
* Top N articles are then filtered from it

Learning about how TF-IDF works has been the most interesting part of this project, giving me insights as to how many NLP applications are based upon it. While it was mostly fun to bring this project to life, the hardest part was to understand how to save the preprocessed model and the matrix and finally apply it as needed when the user invokes the search query.

### Improvement

Looking back, there are a lot of improvements that can be done in this project, namely:

* Extending this project to more than arXiv papers
* Generating a graph to show the connection lengths based on their similarity scores
* Considering the title of the article along with the abstract
* Using the categories as another dimension to sort the generated similar articles
* Advanced filtering and sorting mechanisms for the web application
* Finding similar articles by inputting the link or the paper ID

Also, using better hardware shall help process articles dating back to 20th century providing lots of missing studies that might be relevant to the user. While it takes ~2 seconds to search, it still could be reduced down using advanced caching system at the server side.

## Acknowledgements<a name="ack"></a>

Thanks to [Kaggle](https://www.kaggle.com/), [Cornell University](https://www.kaggle.com/Cornell-University), [arXiv](https://arxiv.org) for providing the dataset and [Udacity](https://www.udacity.com/) for delivering the data science course materials.
