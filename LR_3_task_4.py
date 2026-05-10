import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0
)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії:")
print(regr.coef_)

print("\nВільний член:")
print(regr.intercept_)

print("\nR2 =", round(r2_score(ytest, ypred), 2))
print("MAE =", round(mean_absolute_error(ytest, ypred), 2))
print("MSE =", round(mean_squared_error(ytest, ypred), 2))

plt.scatter(ytest, ypred, edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
plt.xlabel('Виміряно')
plt.ylabel('Передбачено')
plt.title('Регресія для набору diabetes')
plt.show()
