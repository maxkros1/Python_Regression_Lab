import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


iris = load_iris()

X = iris.data
y = iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

clf = RidgeClassifier(tol=1e-2, solver="sag")

clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

print("RIDGE CLASSIFIER FOR IRIS")
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))
print("Precision:", np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print("Recall:", np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print("F1 Score:", np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print("Cohen Kappa Score:", np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print("Matthews Corrcoef:", np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

print("\nClassification Report:")
print(metrics.classification_report(ytest, ypred, target_names=iris.target_names))

mat = confusion_matrix(ytest, ypred)

plt.figure()
plt.imshow(mat)
plt.title("Confusion Matrix — Ridge Classifier")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, mat[i, j], ha="center", va="center")

plt.xticks(np.arange(3), iris.target_names, rotation=45)
plt.yticks(np.arange(3), iris.target_names)

plt.tight_layout()
plt.savefig("Confusion.jpg")
plt.show()
