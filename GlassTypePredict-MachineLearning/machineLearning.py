import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("glass.csv")
X = df.drop("Type", axis=1)
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
model = RandomForestClassifier(random_state=9)
model.fit(X_train, y_train)
cam_pred = model.predict(X_test)

accuracy_score(y_test, cam_pred)
print(classification_report(y_test, cam_pred))
