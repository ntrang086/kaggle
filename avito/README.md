[//]: # (Image References)

[image1]: ./dataset.png "Product description"
[image2]: ./dataset2.jpg "Data file relations"

# Avito Demand Prediction Challenge

## Introduction

When selling used goods online, a combination of tiny, nuanced details in a product description can make a big difference in drumming up interest. Details like:
![Product description][image1]

Avito, Russia’s largest classified advertisements website, is deeply familiar with this problem. Sellers on their platform sometimes feel frustrated with both too little demand (indicating something is wrong with the product or the product listing) or too much demand (indicating a hot item with a good description was underpriced).

In [their fourth Kaggle competition](https://kaggle.com/c/avito-demand-prediction), [Avito](https://www.avito.ru/) is challenging participants to predict demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. With this information, Avito can inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive.

## Approach

In this project, we will work with and engineer different types of data: images, text, categorical and numerical data. We will train a [LightGBM](https://github.com/Microsoft/LightGBM) model to predict deal probabilities. LightGBM is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms. It is under the umbrella of the [DMTK](http://github.com/microsoft/dmtk) project of Microsoft.

## Code

* `extract_image_scores.ipynb` - Extracts image classification scores using a pre-trained model.
* `aggregated_features.ipynb` - Engineers aggregated features using `train_active.csv`, `test_active.csv`, `periods_test.csv` and `periods_test.csv`.
* `lightGBM_model.ipynb`- Engineers the rest of the features and trains a LightGBM model to predict deal probabilities. 
* `utils.py` - Contains helper code for engineering features.

## Setup

You need the following: 

* Python 3
* lightgbm
* nltk
* numpy
* pandas
* keras with tensorflow backend
* sklearn
* matplotlib
* matplotlib_venn

## Data

The data comes from the Kaggle's Avito competition [page](https://www.kaggle.com/c/avito-demand-prediction/data). Here's a summary of their relations:

![Data file relations][image2]

1) Download the following csv files and put them in a `csv` subdirectory:

* `train.csv` - Train data.
* `test.csv` - Test data. Same schema as the train data, minus deal_probability.
* `train_active.csv` - Supplemental data from ads that were displayed during the same period as train.csv. Same schema as the train data, minus deal_probability.
* `test_active.csv` - Supplemental data from ads that were displayed during the same period as test.csv. Same schema as the train data, minus deal_probability.
* `periods_train.csv` - Supplemental data showing the dates when the ads from train_active.csv were activated and when they where displayed.
* `periods_test.csv` - Supplemental data showing the dates when the ads from test_active.csv were activated and when they where displayed. Same schema as periods_train.csv, except that the item ids map to an ad in test_active.csv.


2) Download the following image zipped folders and put them in `images/train` and `images/test` subdirectories respectively:
* `train_jpg.zip` - Images from the ads in train.csv.
* `test_jpg.zip` - Images from the ads in test.csv.

3) Other:
* `sample_submission.csv` - A sample submission in the correct format.
          
## Run

To run any script file, use:

```bash
python <script.py>
```

To run any IPython Notebook, use:

```bash
jupyter notebook <notebook_name.ipynb>
```
