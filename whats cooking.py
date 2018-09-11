# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:54:07 2018

@author: vibhanshu vaibhav
"""
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
cv=CountVectorizer(binary='true')

df=pd.read_json('train.json',orient='columns')
test_df=pd.read_json('test.json',orient='columns')

cuisines={'thai':0,'vietnamese':1,'spanish':2,'southern_us':3,'russian':4,'moroccan':5,'mexican':6,'korean':7,'japanese':8,'jamaican':9,'italian':10,'irish':11,'indian':12,'greek':13,'french':14,'filipino':15,'chinese':16,'cajun_creole':17,'british':18,'brazilian':19 }
df.cuisine= [cuisines[item] for item in df.cuisine]

ho=df['ingredients']
hs=test_df['ingredients']

def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

p0= re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
ho= sub_match(p0, ' ', ho)
# remove all digits
p1 = re.compile(r'\d+')
ho = sub_match(p1, ' ', ho)
# remove all the non-letter characters
p2 = re.compile('[^\w]')
ho= sub_match(p1, ' ', ho)

p4= re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
hs= sub_match(p4, ' ', hs)
# remove all digits
p5 = re.compile(r'\d+')
hs = sub_match(p5, ' ', hs)
# remove all the non-letter characters
p6 = re.compile('[^\w]')
hs= sub_match(p6, ' ', hs)

df['n']=ho
test_df['m']=hs

df['seperated_ingredients'] = df['n'].apply(','.join)
test_df['seperated_ingredients'] = test_df['m'].apply(','.join)

df['seperated_ingredients']=df['seperated_ingredients'].str.lower()
test_df['seperated_ingredients']=test_df['seperated_ingredients'].str.lower()

h1=df['seperated_ingredients']
h2=test_df['seperated_ingredients']

h1=h1.replace('((www\.[\s]+)|(https?://[^\s]+))','URL',regex=True)
h1=h1.replace(r'#([^\s]+)', r'\1', regex=True)
h1=h1.replace('\'"',regex=True)

h1=tv.fit_transform(h1)

h2=h2.replace('((www\.[\s]+)|(https?://[^\s]+))','URL',regex=True)
h2=h2.replace(r'#([^\s]+)', r'\1', regex=True)
h2=h2.replace('\'"',regex=True)
h2=tv.transform(h2)

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(h1,df['cuisine'], random_state=0)
classifier=BernoulliNB().fit(X_train,y_train)
classifier.score(X_test,y_test)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
neigh.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(penalty='l1')
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

from sklearn.linear_model import LogisticRegression
clf1= LogisticRegression(penalty='l1')
clf1.fit(h1,df['cuisine'])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

classifier = SVC(C=100, # penalty parameter, setting it to a larger value 
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value, not tuned yet
                 gamma=1, # kernel coefficient, not tuned yet
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)

model = OneVsRestClassifier(classifier, n_jobs=1)
model.fit(X_train,y_train)
model.score(X_test,y_test)

ss=model.predict(hs)


col=['id','cuisine']
df_csv = pd.DataFrame(columns=col)
df_csv.id=test_df.id
df_csv.cuisine=ss
df_csv.to_csv('sample_submission.csv',index=False)

