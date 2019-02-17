import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

# Define 2 classes and their Features
dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnngs.warn("K is set to a bad value")
    distances = []
    for group in data:
        for features in data[group]:
            dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([dist, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

result, confidence = k_nearest_neighbors(dataset, new_features, k=3)
print(result, confidence)

#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], s=100, color=result)
#plt.show()

# Load dataset
df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace("?", -99999, inplace=True)
print(df.head())
df.drop(['id'], 1, inplace=True)
fulldata = df.astype('float').values.tolist()
random.shuffle(fulldata)

# Separate training and testing data
test_size = 0.4
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = fulldata[:-int(test_size*len(fulldata))]
test_data = fulldata[-int(test_size*len(fulldata)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(vote, confidence)
        total += 1
accuracy = correct/total
print(accuracy)
