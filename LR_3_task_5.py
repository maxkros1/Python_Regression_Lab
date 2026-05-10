import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

m = 100
X = np.linspace(-3, 3, m)
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m)

X = X.reshape(-1, 1)

linear_model = LinearRegression()
linear_model.fit(X, y)

y_linear_pred = linear_model.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)

print("Перше значення X:")
print(X[0])

print("\nПерше значення X_poly:")
print(X_poly[0])

print("\nЛінійна модель:")
print("intercept =", linear_model.intercept_)
print("coef =", linear_model.coef_)

print("\nПоліноміальна модель:")
print("intercept =", poly_model.intercept_)
print("coef =", poly_model.coef_)

print("\nОцінка лінійної регресії:")
print("MAE =", round(mean_absolute_error(y, y_linear_pred), 2))
print("MSE =", round(mean_squared_error(y, y_linear_pred), 2))
print("R2 =", round(r2_score(y, y_linear_pred), 2))

print("\nОцінка поліноміальної регресії:")
print("MAE =", round(mean_absolute_error(y, y_poly_pred), 2))
print("MSE =", round(mean_squared_error(y, y_poly_pred), 2))
print("R2 =", round(r2_score(y, y_poly_pred), 2))

plt.scatter(X, y, color='blue', label='Вихідні дані')
plt.plot(X, y_linear_pred, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='green', linewidth=2, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Варіант 10: лінійна та поліноміальна регресія')
plt.legend()
plt.show()
