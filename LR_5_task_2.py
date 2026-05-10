import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
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


input_file = 'data_imbalance.txt'

data = np.loadtxt(input_file, delimiter=',')
X = data[:, :-1]
y = data[:, -1].astype(int)

class_0 = X[y == 0]
class_1 = X[y == 1]

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], marker='x', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], marker='o', label='Class 1')
plt.title('Вхідні дані з дисбалансом класів')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.legend()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

models = {
    'Без балансування': ExtraTreesClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=0
    ),
    'З балансуванням class_weight=balanced': ExtraTreesClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=0,
        class_weight='balanced'
    )
}

for name, classifier in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    classifier.fit(X_train, y_train)

    visualize_classifier(classifier, X_train, y_train, name + ' — навчальна вибірка')

    y_test_pred = classifier.predict(X_test)

    visualize_classifier(classifier, X_test, y_test, name + ' — тестова вибірка')

    print("\nРезультати на навчальній вибірці:")
    print(classification_report(y_train, classifier.predict(X_train), zero_division=0))

    print("\nРезультати на тестовій вибірці:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

plt.show()
