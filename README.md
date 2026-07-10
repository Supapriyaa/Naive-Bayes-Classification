import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_excel("Iris_25_Dataset.xlsx")

print(df.head())

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values")
print(y_test.values)

print("Predicted Values")
print(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

new_data = [[5.8, 2.7, 5.1, 1.9]]

prediction = model.predict(new_data)

print("Predicted Species:", prediction[0])
