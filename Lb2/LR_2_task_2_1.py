import numpy as np

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 8000

with open(input_file, 'r') as f:
    for line in f.readlines():

        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1


X = np.array(X)

X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])

X_data = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=5
)

classifier = SVC(kernel='poly', degree=3)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("SVM POLYNOMIAL KERNEL")
print("Accuracy =", round(accuracy_score(y_test, y_pred), 4))
print("Precision =", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall =", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1 score =", round(f1_score(y_test, y_pred, average='weighted'), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))
