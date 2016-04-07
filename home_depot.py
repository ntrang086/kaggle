import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn import pipeline, grid_search


# Stem words from a string
def stem_word(s, stemmer): 
    if isinstance(s, str):
        return " ".join([stemmer.stem(word) for word in s.lower().split()])
    else:
        return "null"


# Count the number of words two strings have in common
def count_common_word(s1, s2):
    count = 0
    for word in s1.split():
        if s2.find(word) >= 0:
            count += 1
    return count


# Count the number of times the whole s1 (search_term) appears in s2 (product_title or product_description)
def match_whole_word(s1, s2, i):
    count = 0
    while i < len(s2):
        i = s2.find(s1, i)
        if i == -1:
            return count
        else:
            count += 1
            i += len(s1)
    return count


# Process train and test data
def process_data(df, df_prod_desc, df_attribute, df_brand, stemmer):

	# Merge train/test data frame with product description and brand data frames
	df = pd.merge(df, df_prod_desc, how='left', on='product_uid')
	df = pd.merge(df, df_brand, how='left', on='product_uid')
	
	# Stem words
	df['search_term'] = df['search_term'].map(lambda x:stem_word(x, stemmer))
	df['product_title'] = df['product_title'].map(lambda x:stem_word(x, stemmer))
	df['product_description'] = df['product_description'].map(lambda x:stem_word(x, stemmer))
	df['brand'] = df['brand'].map(lambda x:stem_word(x, stemmer))

	# Get the length of search_term, product_title and product_description
	df['search_term_len'] = df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
	df['prod_title_len'] = df['product_title'].map(lambda x:len(x.split())).astype(np.int64)
	df['prod_desc_len'] = df['product_description'].map(lambda x:len(x.split())).astype(np.int64)
	df['brand_len'] = df['brand'].map(lambda x:len(x.split())).astype(np.int64)

	# Count the number of words search_term has in common with product_title and product_description
	df['common_query_title'] = df.apply(lambda x:count_common_word(x['search_term'], x['product_title']), axis=1)
	df['common_query_desc'] = df.apply(lambda x:count_common_word(x['search_term'], x['product_description']), axis=1)

	# Count the number of times the whole search_term appears in product_title and product_description
	df['whole_query_in_title'] = df.apply(lambda x:match_whole_word(x['search_term'], x['product_title'], 0), axis=1)
	df['whole_query_in_desc'] = df.apply(lambda x:match_whole_word(x['search_term'], x['product_description'], 0), axis=1)

	# Calculate the ratio between the number of common words in product_title/product_description and the search_term length
	df['ratio_title'] = df['common_query_title']/df['search_term_len']
	df['ratio_description'] = df['common_query_desc']/df['search_term_len']

	# Count the number of words search_term has in common with brand
	df['common_query_brand'] = df.apply(lambda x:count_common_word(x['search_term'], x['brand']), axis=1)
	df['ratio_brand'] = df['common_query_brand']/df['search_term_len']

	df = df.drop(['search_term','product_title','product_description','brand'],axis=1)
	return df


stemmer = SnowballStemmer('english')

# Read in the data from csv files
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_prod_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
df_attribute = pd.read_csv('attributes.csv', encoding="ISO-8859-1")

# Extract the brand of the product from attributes.csv. df_brand will be merged with train/test data frame
df_brand = df_attribute[df_attribute.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

# Process train and test data
df_train = process_data(df_train, df_prod_desc, df_attribute, df_brand, stemmer)
df_test = process_data(df_test, df_prod_desc, df_attribute, df_brand, stemmer)

id_test = df_test['id']

features_train = df_train.drop(['id','relevance'],axis=1).values
labels_train = df_train['relevance'].values
features_test = df_test.drop(['id'],axis=1).values


"""
# gridsearch to find the best parameters

rf = RandomForestRegressor()
br = BaggingRegressor(rf)

pipe = pipeline.Pipeline([('rf', rf), ('br', br)])

parameters = dict(rf__n_estimators=[5, 10, 15, 20], rf__max_depth=[2, 4, 6, 8, 10], rf__random_state=[0, 5, 10, 15],
	br__n_estimators=[5, 15, 25, 35, 45, 55], br__max_samples=[0.1, 0.2, 0.3], br__random_state=[0, 5, 10, 15, 20, 25, 30])
model = grid_search.GridSearchCV(pipe, parameters)
model.fit(features_train, labels_train)

print("Best parameters:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

#Best parameters:
#{'br__max_samples': 0.1, 'br__n_estimators': 45, 'rf__max_depth': 6, 'br__random_state': 25, 'rf__random_state': 0, 'rf__n_estimators': 5}
#Best CV score: 0.13390585367

pred = model.predict(features_test)
"""

# Use the best parameters from gridsearch
rf = RandomForestRegressor(n_estimators=5, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# Write predicted numbers to submission.csv file
pd.DataFrame({"id": id_test, "relevance": pred}).to_csv('submission.csv',index=False)
