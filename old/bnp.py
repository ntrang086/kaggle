# coding: utf8


"""
https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
Kagglers are challenged to predict the category of a claim based on features available early in the process, 
helping BNP Paribas Cardif accelerate its claims process and therefore provide a better service to its customers.
"""


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# Input csv data as DataFrame
TRAIN_DF = pd.read_csv('train.csv')
TEST_DF = pd.read_csv('test.csv')


# Explore the data
def explore_data(data_frame):
	print data_frame.head()
	print data_frame.info()
	# total number of targets
	total_target = len(data_frame['target'])
	print "total number of claims: ", total_target
	# total number of "1" targets
	one_target = data_frame.loc[data_frame['target'] == 1]['target']
	print "% of claims suitable for accelerated approval: ", float(len(one_target))/total_target


# Deal with missing values
def handle_missing_vals(data_frame):
	count_object_na = 0
	count_nan = 0

	for field in data_frame:
		if data_frame[field].dtype == 'O': # if Series is an object, select the most frequently-occurring element, i.e. index 0
			data_frame[field] = pd.factorize(data_frame[field])[0]
			count_object_na += 1
		else: # find series that have NAN values and replace them with the mean of all values in that series 
			if data_frame[field].isnull().sum() > 0:
				data_frame.loc[data_frame[field].isnull(), field] = data_frame[field].mean()
				count_nan += 1
	
	return data_frame



# Process the train and test data
def process_data(train_df, test_df):
	print "Handle missing values"
	train_df = handle_missing_vals(train_df)
	test_df = handle_missing_vals(test_df)
	return train_df, test_df


# Split the training data set into train and test set for local testing
def split_train_test_from_train(train_df):
	features = train_df.drop(["ID", "target"], axis=1)
	labels = train_df["target"]
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
	return features_train, features_test, labels_train, labels_test


# Split features and labels for actual submission
def split_features_labels(train_df, test_df):
	features_train = train_df.drop(["ID","target"],axis=1)
	labels_train = train_df["target"]
	features_test  = test_df.drop("ID",axis=1).copy()
	return features_train, labels_train, features_test


# Modeling: Try different classifiers

def model(features_train, labels_train, features_test):
	clf = LogisticRegression()
	#clf = RandomForestClassifier()
	clf.fit(features_train, labels_train)
	return clf.predict_proba(features_test)[:,1]


def model_pca_pipeline(features_train, labels_train, features_test):
	clf = LogisticRegression()
	pca = PCA()
	pipe = Pipeline(steps=[('pca', pca), ('logistic', clf)])

	# Plot the PCA spectrum
	pca.fit(features_train)

	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.xlabel('n_components')
	plt.xlim([0,10])
	plt.ylabel('explained_variance_')

	# Prediction
	n_components = [1, 3, 5, 7, 10]

	Cs = np.logspace(-4, 4, 3)

	estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
	estimator.fit(features_train, labels_train)

	plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
	            linestyle=':', label='n_components chosen')
	plt.legend(prop=dict(size=12))
	plt.show()

	return estimator.predict_proba(features_test)[:,1]



def model_xgb(features_train, labels_train, features_test):
	params = { 
   "objective": "binary:logistic",
   "silent": 1,
   "eval_metric": "auc",
   "eta": 0.03, #tried 0.01, 0.03, 0.04, 0.1
   "subsample": 0.5, #Setting it to 0.5 means that XGBoost randomly collected half of
   #the data instances to grow trees and this will prevent overfitting
   "colsample_bytree": 0.7, # tried 0.1, 0.4, 0.6, 0.9, 1
   "max_depth": 7
	}
	train_xgb = xgb.DMatrix(features_train, labels_train)
	test_xgb  = xgb.DMatrix(features_test)
	clf = xgb.train(params, train_xgb, num_boost_round=500) # tried 100, 200, 1000
	return clf.predict(test_xgb)


# Calculate logloss for local testing
# https://www.kaggle.com/wiki/LogarithmicLoss

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# Submit the prediction probability data as a csv file
def submit(test_df, labels_test):
	submit_df = pd.DataFrame()
	submit_df["ID"] = test_df["ID"]
	submit_df["PredictedProb"] = labels_test
	submit_df.to_csv('bnp_ntrang.csv', index=False)



if __name__ == "__main__":
	explore_data(TRAIN_DF)
	print "Process the data"
	train_df, test_df = process_data(TRAIN_DF, TEST_DF)
	print "Split features and labels"
	features_train, labels_train, features_test = split_features_labels(train_df, test_df)
	#features_train, features_test, labels_train, labels_test = split_train_test_from_train(train_df)
	print "Model the data"
	pred = model_xgb(features_train, labels_train, features_test)
	#print logloss(labels_test, pred)
	print "Submmit the data"
	submit(test_df, pred)
	print "Done"
