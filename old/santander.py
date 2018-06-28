"""
https://www.kaggle.com/c/santander-customer-satisfaction
From frontline support teams to C-suites, customer satisfaction is a key measure of success. 
Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.
Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. 
Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied 
with their banking experience.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import normalize, Binarizer, scale
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, f_classif
from sklearn import grid_search
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
import operator
import itertools


TOP_FEATURES = ['saldo_var42', 'num_var22_ult3', 'num_var45_hace2', 'saldo_var5', 'num_var45_ult3', 'num_var45_hace3', 
'saldo_medio_var5_ult1', 'saldo_var30', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
'var15', 'var38']


# Explore the data
def explore_data(str_df_name, df): # str_df_name must contain either 'train' or 'test'
	#print (df.head())
	#print (df.info())
	#print (df.describe())
	
	# Check out the training set
	if 'train' in str_df_name:
		total_target = len(df['TARGET'])
		print ("total number of customers: ", total_target)
		
		one_target = df.loc[df['TARGET'] == 1]['TARGET']
		print ("% of customers that are satisfied: ", float(len(one_target))/total_target)
	
	# If neither 'train' nor 'test' is in the name, do nothing
	elif 'train' not in str_df_name and 'test' not in str_df_name:
		print ("Did not explore", str_df_name, "dataframe. " 
			"Make sure the name you've given to the dataframe contains either 'train' or 'test'")
		return
	
	# Check out the TOP_FEATURES list and see if there is any null value
	count_nan = 0
	for field in df:
		if field in TOP_FEATURES:
			print (df[field].describe())
			#df[field].hist(bins=100)
			#plt.gcf().savefig(field)
		if df[field].isnull().sum() > 0:
			count_nan += 1
	
	print ("Number of NaN values:", count_nan)
	

	# Explore features that have constant values detected by SelectPercentile (f_classif)
	print (df.ix[:,349].describe())

	

"""
Observations from data exploration
- All series are either int or float
- No missing value
- var3 suspected to be the nationality of a customer: min value is -999999
- var38 is suspected to be the mortage value with the bank
- var15 suspected to be the age of a customer
- Some fields have 9999999999, which can represent a missing value
- SelectPercentile (f_classif) identifies the following constant features:
[ 21  22  56  57  58  59  80  84  85 131 132 133 134 155 161 162 179 180
 189 192 220 222 234 238 244 248 261 262 303 307 315 319 327 349] are constant.
"""


# Process the train and test data
def process_data(df):

	# Replace -999999 value in var3 with the most frequently occuring value
	# Use .loc[row_index,col_indexer] = value to avoid SettingWithCopyWarning
	df.loc[df['var3'] == -999999, 'var3'] = df['var3'].value_counts().idxmax()
	
	# Replace 9999999999 with the most frequently occuring value
	# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
	# Also replace NA values if any
	for field in df:
		df.loc[df[field] == 9999999999, field] = df[field].value_counts().idxmax()
		df.loc[df[field].isnull(), field] = df[field].value_counts().idxmax()

	# Drop rows and columns that have ALL NA values
	df.dropna(axis=0, how='all', inplace=True)
	df.dropna(axis=1, how='all', inplace=True)

	#df.to_csv('df.csv', index=False)

	return df


# Split the training data set into train and test set for local testing
def split_train_test_from_train(df_train):
	features = df_train.drop(["ID", "TARGET"], axis=1)
	labels = df_train["TARGET"]
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
	return features_train, features_test, labels_train, labels_test


# Split features and labels for actual submission
def split_features_labels(df_train, df_test):
	features_train = df_train.drop(["ID","TARGET"],axis=1)
	labels_train = df_train["TARGET"]
	features_test  = df_test.drop("ID",axis=1).copy()
	return features_train, labels_train, features_test


# Modeling: xgboost outperforms LogisticRegression and RandomForest
# Combination of chi2 and f_classif performs better than each individual method
# Nice to try: http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py

def model_xgb(features_train, labels_train, features_test):
	
	# Remove constant features
	selector_vt = VarianceThreshold()
	selector_vt.fit(features_train)

	# Get the indices of zero variance features
	features_kept = selector_vt.get_support(indices=True)
	orig_features = np.arange(features_train.columns.size)
	features_deleted = np.delete(orig_features, features_kept)
	#print ("Indices of deleted features:", features_deleted)
	print ("- Number of constant features removed:", len(features_deleted))

	# Delete zero variance features from train and test sets
	features_train = features_train.drop(labels=features_train.columns[features_deleted], axis=1)
	features_test = features_test.drop(labels=features_test.columns[features_deleted], axis=1)
	#print (features_train.shape, features_test.shape)


	"""
	# Another way of removing constant features. Slightly slower than the above method
	# Count the number of unique values in each feature
	nuniques_train = features_train.apply(lambda x:x.nunique())
	no_variation_train = nuniques_train[nuniques_train==1].index
	features_train = features_train.drop(no_variation_train, axis=1)

	features_test = features_test.drop(no_variation_train, axis=1)
	print (features_train.shape, features_test.shape)
	"""
	
	# Remove idential features
	features_deleted = []
	
	# Find the names of identical features by going through all the combinations of features
	for f1, f2 in itertools.combinations(iterable=features_train.columns, r=2):
		if np.array_equal(features_train[f1], features_train[f2]):
			features_deleted.append(f2)
	features_deleted = np.unique(features_deleted)

	# Delete the identical features
	features_train = features_train.drop(labels=features_deleted, axis=1)
	features_test = features_test.drop(labels=features_deleted, axis=1)
	print ("- Number of idential features removed:", len(features_deleted))

	# Add a column to count the number of zeros per row
	features_train['n0'] = (features_train == 0).sum(axis=1)
	features_test['n0'] = (features_test == 0).sum(axis=1)


	# Feature normalization
	f_train_normalized = normalize(features_train, axis=0)
	f_test_normalized = normalize(features_test, axis=0)

	# Do PCA
	print ("- Do PCA")
	pca = PCA(n_components=2)
	f_train_pca = pca.fit_transform(f_train_normalized)
	features_train['PCA1'] = f_train_pca[:,0]
	features_train['PCA2'] = f_train_pca[:,1]
	
	f_test_pca = pca.fit_transform(f_test_normalized)
	features_test['PCA1'] = f_test_pca[:,0]
	features_test['PCA2'] = f_test_pca[:,1]

	# Feature selection
	#p = 75, AUC = 0.834348
	p = 70 # AUC = 0.834820
	#p = 65, AUC = 
	print ("- Do feature selection")
	f_train_binarized = Binarizer().fit_transform(scale(features_train))
	selector_chi2 = SelectPercentile(chi2, percentile=p).fit(f_train_binarized, labels_train)
	selected_chi2 = selector_chi2.get_support() # a list of True/False to indicate if a feature is selected or not
	#selected_chi2_features = [f for i, f in enumerate(features_train.columns) if selected_chi2[i]]
	#print (selected_chi2_features)

	select_f_classif = SelectPercentile(f_classif, percentile=p).fit(f_train_binarized, labels_train)
	selected_f_classif = select_f_classif.get_support() # a list of True/False to indicate if a feature is selected or not
	#selected_f_classif_features = [f for i, f in enumerate(features_train.columns) if selected_f_classif[i]]
	#print (selected_f_classif_features)

	selected = selected_chi2 & selected_f_classif
	selected_features = [f for i, f in enumerate(features_train.columns) if selected[i]]
	#print (selected_features)

	features_train = features_train[selected_features]
	features_test = features_test[selected_features]

	
	# xgboost
	print ("- Perform xgboost")
	params = { 
	"objective": "binary:logistic",
	"silent": 1,
	"eval_metric": "auc",
	"eta": 0.03, # tried 0.01
	"subsample": 0.5, # tried 1.0, 0.4
	"colsample_bytree": 0.7, # tried 0.5, 0.9
	"max_depth": 2  # 2-->AUC=0.836347; 5 --> AUC=0.835131; 7 -> AUC=0.834351
	#"min_child_weight": 1, # tried 2 & 5
	#"gamma": 0 # tried 4
	}

	train_xgb = xgb.DMatrix(features_train, labels_train)
	test_xgb  = xgb.DMatrix(features_test)
	clf = xgb.train(params, train_xgb, num_boost_round=500) # tried 400, 500, 600

	# Get the importances of features, returning pairs of features and their importances
	importance = clf.get_fscore() 

	# Sort features by importance, and return the top features only
	# 'key' parameter specifies a function to be called on each list element prior to making comparisons
	# itemgetter(1) returns importances, itemgetter(0) returns features
	sorted_importance = sorted(importance.items(), key=operator.itemgetter(1))[-15:]
	#print (sorted_importance)

	# Put pairs of features and their importances into a DataFrame for plotting
	df_importance = pd.DataFrame(sorted_importance, columns=['feature', 'fscore'])

	# Plot the importance of features, which is useful for data exploration phase
	df_importance.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 6))
	plt.title('XGBoost Feature Importance')
	plt.xlabel('feature importance')
	plt.gcf().savefig('feature_importance_xgb.png')
	#plt.show() # if putting show() before gcf().savefig, the figure won't be saved

	return clf.predict(test_xgb)
	

# xgboost with GridSearch. Not able to run, too slow
def model_xgb_grid_search(features_train, labels_train, features_test):
	
	# Feature normalization
	f_train_normalized = normalize(features_train, axis=0)
	f_test_normalized = normalize(features_test, axis=0)

	# Do PCA
	pca = PCA(n_components=2)
	f_train_pca = pca.fit_transform(f_train_normalized)
	features_train['PCA1'] = f_train_pca[:,0]
	features_train['PCA2'] = f_train_pca[:,1]
	
	f_test_pca = pca.fit_transform(f_test_normalized)
	features_test['PCA1'] = f_test_pca[:,0]
	features_test['PCA2'] = f_test_pca[:,1]

	# Feature selection
	#p = 75, AUC = 0.822000
	p = 70 # AUC = 0.833136
	#p = 65, AUC = 0.832403
	f_train_binarized = Binarizer().fit_transform(scale(features_train))
	select = SelectPercentile(chi2, percentile=p).fit(f_train_binarized, labels_train)
	selected = select.get_support() # a list of True/False to indicate if a feature is selected or not
	selected_features = []
	for i,f in enumerate(features_train.columns):
		if selected[i]:
			selected_features.append(f)
	#print (selected_features)

	features_train = features_train[selected_features]
	features_test = features_test[selected_features]

	# xgboost with GridSearch
	
	xgb_mobdel = XGBClassifier()
	params = {
	"max_depth": [2, 5, 10],
	"min_child_weight": [1, 2, 6]
	}
	clf = GridSearchCV(estimator=xgb_mobdel, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
	clf.fit(features_train, labels_train)
	print (clf.grid_scores_, clf.best_params_, clf.best_score_)

	# Get the importances of features, returning pairs of features and their importances
	importance = clf.get_fscore() 

	# Sort features by importance, and return the top features only
	# 'key' parameter specifies a function to be called on each list element prior to making comparisons
	# itemgetter(1) returns importances, itemgetter(0) returns features
	sorted_importance = sorted(importance.items(), key=operator.itemgetter(1))[-15:]
	print (sorted_importance)

	# Put pairs of features and their importances into a DataFrame for plotting
	df_importance = pd.DataFrame(sorted_importance, columns=['feature', 'fscore'])

	# Plot the importance of features, which is useful for data exploration phase
	df_importance.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 6))
	plt.title('XGBoost Feature Importance')
	plt.xlabel('feature importance')
	plt.gcf().savefig('feature_importance_xgb.png')
	#plt.show() # if putting show() before gcf().savefig, the figure won't be saved

	return clf.predict(test_xgb)



# Submit the prediction probability data as a csv file
def submit(df_test, labels_test):
	submit_df = pd.DataFrame()
	submit_df["ID"] = df_test["ID"]
	submit_df["TARGET"] = labels_test
	submit_df.to_csv('st_submission.csv', index=False)



if __name__ == "__main__":
	# Input csv data as DataFrame
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	print ("Explore the training data")
	explore_data("train_data", df_train)
	print ("Explore the testing data")
	explore_data("test_data", df_test)
	
	print ("Process the data")
	df_train = process_data(df_train)
	df_test  = process_data(df_test)
	
	print ("Split features and labels")
	features_train, labels_train, features_test = split_features_labels(df_train, df_test)
	#features_train, features_test, labels_train, labels_test = split_train_test_from_train(df_train)
	print ("Model the data")
	pred = model_xgb(features_train, labels_train, features_test)

	# AUC for local testing only
	#print('\nOverall AUC:', roc_auc_score(labels_test, pred))

	print ("Submmit the data")
	submit(df_test, pred)
	print ("Done")