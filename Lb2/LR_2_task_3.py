import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


iris_dataset = load_iris()

print("Ключі iris_dataset:")
print(iris_dataset.keys())

print("\nОпис набору даних:")
print(iris_dataset['DESCR'][:700])

print("\nНазви класів:")
print(iris_dataset['target_names'])

print("\nНазви ознак:")
print(iris_dataset['feature_names'])

print("\nТип масиву data:")
print(type(iris_dataset['data']))

print("\nФорма масиву data:")
print(iris_dataset['data'].shape)

print("\nПерші 5 рядків даних:")
print(iris_dataset['data'][:5])

print("\nФорма масиву target:")
print(iris_dataset['target'].shape)

print("\nМітки класів:")
print(iris_dataset['target'])


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nKNN CLASSIFIER FOR IRIS")
print("Accuracy =", round(accuracy_score(y_test, y_pred), 4))
print("Precision =", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall =", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1 score =", round(f1_score(y_test, y_pred, average='weighted'), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris_dataset['target_names']))

X_new = np.array([[5.0, 2.9, 1.0, 0.2]])

print("\nФорма масиву X_new:")
print(X_new.shape)

prediction = knn.predict(X_new)

print("Прогноз:")
print(prediction)

print("Спрогнозована мітка:")
print(iris_dataset['target_names'][prediction][0])
