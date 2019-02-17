import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd

# Load data
df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace("?", 99999, inplace=True)
print(df.head())
#df.drop(['id'], 1, inplace=True)

# Features (X) and label (y)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Seperate training and testing data
X_train, X_test, y_train, y_test = \
model_selection.train_test_split(X, y, test_size=0.2)

# Make classifiers
clf = svm.SVC()
clf.fit(X_train, y_train)

# Score the classifiers
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1]).reshape(1, -1)

prediction = clf.predict(example_measure)
print(prediction)
