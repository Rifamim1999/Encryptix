#import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

## Read the csv file
dataset = pd.read_csv(r'F:\INTERNSHIP\Encryptix\spam.csv', encoding='latin1')

## Finding the independent and dependent variable
X = dataset.iloc[:, 1].values   
y = dataset.iloc[:, 0].values   
print(X)
print(y)

## Spliting the dataset into test and train set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
print(X_train)

## Finding the TfIdf vector
tfidf_vector = TfidfVectorizer(stop_words=['I', 'am', 'is', 'you', 'your', 'the'], max_df=0.8)
X_train = tfidf_vector.fit_transform(X_train)
X_test = tfidf_vector.transform(X_test)

## Training using Naive Bayes classifier
naive_model = MultinomialNB()
naive_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_predictions = naive_model.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)*100
print(f"Accuracy using Naive Bayes classifier: {accuracy}%")

# Classification using RandomForest
rf_model = RandomForestClassifier(n_estimators=15, random_state=0)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Random Forest Accuracy: {accuracy}%")

## Classification using SVR
svr_model = SVC()
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f'Accuracy using SVR classifier: {accuracy}%')
