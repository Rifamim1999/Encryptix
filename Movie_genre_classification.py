#importing important libaries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

## Read the the files from the directory
train = pd.read_csv(r'F:\INTERNSHIP\task1\train_data.txt', delimiter=' ::: ', engine='python', header=None)
test = pd.read_csv(r'F:\INTERNSHIP\task1\test_data.txt', delimiter=' ::: ', engine='python', header=None)
test_solve = pd.read_csv(r'F:\INTERNSHIP\task1\test_data_solution.txt', delimiter=' ::: ', engine='python', header=None)

## printing the coloumns
print(train.columns)
print(test.columns)
print(test_solve.columns)

### 0 --> ID, 1 --> Title, 2 --> Genre, 3 --> Description for train and test_solve file
### 0--> ID, 1 --> Title, 2 --> Description

#### Need to make all the words in the description either small letters or capital letters and also need to remove the signs that are not letters

train[3] = train[3].str.lower().str.replace(r'[^\w\s]', '', regex= True)
test[2] = test[2].str.lower().str.replace(r'[^\w\s]', '', regex= True)
test_solve[3] = test_solve[3].str.lower().str.replace(r'[^\w\s]', '', regex=True)


## Creating independent and dependent variables of train and test 
X_train = train[3]
y_train = train[2]
X_test = test[2]
y_test = test_solve[2]

### Finding the TFIDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=['english', 'am', 'is', 'you'], max_df=0.8)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Classification using RandomForest
rf_model = RandomForestClassifier(n_estimators=15, random_state=0)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Random Forest Accuracy: {accuracy}%")

#### Naive Bayes classifier 
### Classification with Naive Bayes
naive_classifiier = MultinomialNB()
naive_classifiier.fit(X_train, y_train)
y_pred = naive_classifiier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f"Accuracy for Naive Bayes Classifier: {accuracy}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

