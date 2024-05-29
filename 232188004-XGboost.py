import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb

df = pd.read_csv('232188004.csv')

df = df.drop(['e_cblp','e_cp','e_cparhdr','e_maxalloc','e_sp','e_lfanew', 'NumberOfSections', 'CreationYear'], axis=1)

df.fillna(0, inplace=True)

le = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

X = df.drop(['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred, zero_division=1))

print("Confusion Matrix:")
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
