import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split


input_file = 'traffic_data.txt'

data = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')

        if len(items) == 5:
            data.append(items)

data = np.array(data)

label_encoders = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
        label_encoders.append(None)
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(data[:, i])
        label_encoders.append(encoder)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

regressor = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=4,
    random_state=0
)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("EXTRA TREES REGRESSOR")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean squared error =", round(mean_squared_error(y_test, y_pred), 2))
print("Explained variance score =", round(explained_variance_score(y_test, y_pred), 2))

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']

test_datapoint_encoded = []

for i, item in enumerate(test_datapoint):
    if label_encoders[i] is None:
        test_datapoint_encoded.append(int(item))
    else:
        test_datapoint_encoded.append(
            int(label_encoders[i].transform([item])[0])
        )

test_datapoint_encoded = np.array(test_datapoint_encoded).reshape(1, -1)

predicted_traffic = regressor.predict(test_datapoint_encoded)[0]

print("\nТестова точка:")
print(test_datapoint)

print("Передбачена кількість машин:", int(predicted_traffic))
