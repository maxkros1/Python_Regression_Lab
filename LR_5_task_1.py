import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report


def visualize_classifier(classifier, X, y, title):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.title(title)
    plt.xlabel('Ознака 1')
    plt.ylabel('Ознака 2')


input_file = 'data_random_forests.txt'

data = np.loadtxt(input_file, delimiter=',')
X = data[:, :-1]
y = data[:, -1].astype(int)

class_0 = X[y == 0]
class_1 = X[y == 1]
class_2 = X[y == 2]

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], marker='s', edgecolors='black', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], marker='o', edgecolors='black', label='Class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], marker='^', edgecolors='black', label='Class 2')
plt.title('Вхідні дані')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.legend()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

params = {
    'n_estimators': 100,
    'max_depth': 4,
    'random_state': 0
}

models = {
    'Random Forest': RandomForestClassifier(**params),
    'Extra Trees': ExtraTreesClassifier(**params)
}

test_datapoints = np.array([
    [5, 5],
    [3, 6],
    [6, 4],
    [7, 2],
    [4, 4],
    [5, 2]
])

for name, classifier in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    classifier.fit(X_train, y_train)

    visualize_classifier(classifier, X_train, y_train, name + ' — навчальна вибірка')

    y_test_pred = classifier.predict(X_test)

    visualize_classifier(classifier, X_test, y_test, name + ' — тестова вибірка')

    print("\nРезультати на навчальній вибірці:")
    print(classification_report(y_train, classifier.predict(X_train)))

    print("\nРезультати на тестовій вибірці:")
    print(classification_report(y_test, y_test_pred))

    print("\nОцінка довіри для тестових точок:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = np.argmax(probabilities)

        print("\nТочка:", datapoint)
        print("Передбачений клас:", predicted_class)
        print("Ймовірності:", probabilities)

    visualize_classifier(
        classifier,
        test_datapoints,
        np.zeros(len(test_datapoints)),
        name + ' — тестові точки'
    )

plt.show()
