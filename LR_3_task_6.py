import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

m = 100
X = np.linspace(-3, 3, m)
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m)

X = X.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    train_errors = []
    val_errors = []

    for m in range(2, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальна вибірка")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Валідаційна вибірка")
    plt.xlabel("Кількість навчальних прикладів")
    plt.ylabel("RMSE")
    plt.title("Криві навчання")
    plt.legend()
    plt.show()

linear_regression = LinearRegression()
plot_learning_curves(linear_regression, X_train, y_train, X_val, y_val)

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(polynomial_regression, X_train, y_train, X_val, y_val)
