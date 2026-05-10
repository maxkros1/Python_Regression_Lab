import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 10000

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

models = {
    'Linear SVM': LinearSVC(random_state=0, max_iter=10000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=10000)
}

print("ПОРІВНЯННЯ КЛАСИФІКАТОРІВ ДЛЯ income_data.txt")
print("-" * 70)

best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n", name)
    print("Accuracy =", round(accuracy, 4))
    print("Precision =", round(precision, 4))
    print("Recall =", round(recall, 4))
    print("F1 score =", round(f1, 4))

    if f1 > best_f1:
        best_f1 = f1
        best_model = name

print("\nНайкращий класифікатор за F1-score:", best_model)
print("Найкращий F1-score:", round(best_f1, 4))
