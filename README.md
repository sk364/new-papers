# New Papers
A recommendation and categorization application to suggest related research papers, given an abstract.

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
* After running setup, the classification report is available to analyze in the console.
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

This project attempts to solve the problem of sifting through a myriad number of articles to filter the relevant articles for the study, by exposing a web application in which the user can provide an abstract of an article to get a list of related articles and the categories associated to it. The application uses the [arXiv research paper dataset](https://www.kaggle.com/Cornell-University/arxiv) to perform knowledge-based recommendations.

To provide a brief summary of the underlying approach taken in this project, the solution uses the TF-IDF (term frequency-inverse document frequency) transformation on the tokens, reflecting the importance of the word in the abstract with respect to the entire collection of abstracts. A one v/s rest multilabel support vector classifier is then used to output the categories associated with the abstract. As to finding the recommendations, cosine similarity technique, taking a cosine angle between the two vectors, is used to find document similarity using the TF-IDF vector of each document, where greater the angle, better the similarity.

## Problem Statement<a name="statement"></a>

The goal of the project is to create a web application to help users find similar articles and the associated categories, given a piece of text using TF-IDF transformations, linear kernel bound support vector classifier and cosine similarity techniques. The application is expected to be useful for conducting literature reviews, finding correlation between past researches and the user's own research, etc.

## Metrics<a name="metrics"></a>

The project evaluates the model on `precision`, `recall` and `f1 score`, where

* Precision = True Positives / (True Positives + False Positives)
* Recall = True Positives / (True Positives + False Negatives)
* F1-score = (2 * Precision * Recall) / (Precision + Recall)

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

**Fig. 3** A plot showing the number of submissions by submission date.

The above figure indicates that the submissions are skewed towards the recent dates.

![screenshot](./assets/category.png?raw=true "Fig. 4")

**Fig. 4** A plot showing the percentage of papers by general category, as observed in a sample of 10,000 articles

Most number of papers are tagged with Computer Science and Mathematics by a really great margin, followed by Condensed Matter and Physics, as shown in Fig. 4.

## Methodology<a name="meth"></a>

### Data Preprocessing & Implementation

In the setup phase of the application, the following steps are executed for the recommendations:

1. Load dataset
2. Filter the data limiting to articles published after the year 2019 (at most 100,000 papers)
3. Transform the data to add in general category
4. Use general category to add dummy columns
5. Compute and save the TF-IDF vectorizer and matrix built using the abstracts
6. Save the dataset

Next, to predict categories, the following steps are executed:

1. Split dataset into training and test sets
2. Build a LinearSVC classifier pipeline to predict categories using OneVsRest classifier
3. Use grid search CV technique to fit the model on the best parameters
4. Save the model

This completes the preprocessing of the data. Finally, the index route can be loaded in the browser to test the application out. Provide an abstract in the input field and click on "Search" to get the similar article recommendations and the predicted categories of the text. This is done sequentially as follows:

1. Load the saved dataset, vectorizer and the TF-IDF matrix
2. Build another vectorizer using the saved vectorizer's vocabulary
3. Compute TF-IDF matrix for the abstract
4. Using cosine similarity method, build a list of indices sorted by best match
5. Fetch metadata from saved dataset and send back the list to the web server
6. Load the saved model
7. Make predictions
8. Send back the list of categories associated with the abstract

The response then contains the similar articles list and the categories list which then is compiled using JavaScript to display it on the web page.

### Refinement

[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) is used for hypertuning the parameters to build the model with the best parameters. The list of parameters used are as following:

* tfidf__use_idf: Use Inverse-Document Refetching while TF-IDF transformation or not
* vect__ngram_range: If words will be unigrams or bigrams, etc or a hybrid mix of n-grams.
* vect__max_df: while building vocabulary, the score of terms higher than this will be ignored
* vect__max_features: capping the number of features
* clf__estimator__C: regularization parameter

The following parameters grid is used:

```python
# python
{
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    'clf__estimator__C': [1, 10, 100, 1000]
}
```

## Results<a name="res"></a>

### Model Evaluation and Validation

To categorize the abstract, a Linear SVC model is used where the grid search CV technique found the best parameters as following:

* Use TF-IDF transformer's Inverse-Document frequency reweighting (param: `use_idf` = True)
* Set Linear SVC classifier's regularization parameter `C` to 100.
* Ignore TF-IDF score less than 0.5.
* Set parameter `ngram_range` to (1, 2), considering both unigrams and bigrams.

The model is split into training and testing set. The latter is then used to validate the model, computing the precision, recall and f1-score. The metrics associated with some categories are as following:

```
1. Category: cs
              precision    recall  f1-score   support

           0       0.95      0.92      0.93      1212
           1       0.88      0.92      0.90       788

    accuracy                           0.92      2000
   macro avg       0.91      0.92      0.92      2000
weighted avg       0.92      0.92      0.92      2000
--

2. Category: math
              precision    recall  f1-score   support

           0       0.94      0.95      0.95      1451
           1       0.87      0.83      0.85       549

    accuracy                           0.92      2000
   macro avg       0.91      0.89      0.90      2000
weighted avg       0.92      0.92      0.92      2000
--

3. Category: physics
              precision    recall  f1-score   support

           0       0.94      0.98      0.96      1791
           1       0.68      0.45      0.54       209

    accuracy                           0.92      2000
   macro avg       0.81      0.71      0.75      2000
weighted avg       0.91      0.92      0.91      2000
```


**Fig. 5** Classification reports for categories math, physics and statistics.

Percentage of articles tagged in actual test set and predicted values have pretty much the same shape and values as shown in Fig. 6 and Fig. 7.

![screenshot](./assets/results-actual.png?raw=true "Fig. 6")

**Fig. 6** Bar graph showing percentage of articles tagged with the 'actual' categories

![screenshot](./assets/results-pred.png?raw=true "Fig. 7")

**Fig. 7** Bar graph showing percentage of articles tagged with the 'predicted' categories

### Justification

The project is able to successfully implement text mining using TF-IDF and Linear SVC to categorize over multiple categories. While the precision, recall and f1-score for each category was greater than 90%, there is still room for improvement.

It is also evident that the application is useful to find similar articles in a clean and compact list view. But, it is limited to arXiv papers with a smaller dataset size. It also has limitations to read in the mathJax format to fully comprehend the ignorance of important mathematical texts.

All in all, the project brings out its usefulness in a limited domain, but a better infrastructure and upgradation of techniques along with a better webview shall provide a really good enhancement to find interesting papers.

## Conclusion<a name="conclusion"></a>

### Reflection

Steps followed to find the similar articles:

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

Steps followed to predict the categories:

* Load dataset
* Split it into train-test sets
* Build a Linear SVC classifier
* Fit the grid search CV model on training set with the set of parameters
* Save the resulting model
* Load the saved model on event of new abstract
* Predict using the model

Learning about how TF-IDF works has been the most interesting part of this project, giving me insights as to how many NLP applications are based upon it. While it was mostly fun to bring this project to life, the hardest part was to understand how to save the preprocessed model and the matrix and finally apply it as needed when the user invokes the search query.

### Improvement

Looking back, there are a lot of improvements that can be done in this project, namely:

* Extending this project to more than arXiv papers
* Generating a graph to show the connection lengths based on their similarity scores
* Enhance web application to take in user ratings over the search results
* In case arXiv and other publications are used, then LDA scheme can be considered for topic modelling
* Considering the title of the article along with the abstract
* Using the categories as another dimension to sort the generated similar articles
* Advanced filtering and sorting mechanisms for the web application
* Finding similar articles by inputting the link or the paper ID

Also, having more memory and better processing power shall help process articles dating back to 20th century providing lots of missing studies that might be relevant to the user.

## Acknowledgements<a name="ack"></a>

Thanks to [Kaggle](https://www.kaggle.com/), [Cornell University](https://www.kaggle.com/Cornell-University), [arXiv](https://arxiv.org) for providing the dataset and [Udacity](https://www.udacity.com/) for delivering the data science course materials.
