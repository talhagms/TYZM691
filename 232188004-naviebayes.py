import pandas as pd

df=pd.read_csv('232188004.csv')
df=df.drop(['e_cblp','e_cp','e_cparhdr','e_maxalloc','e_sp','e_lfanew'],axis=1)

df.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder

df=df.drop(['NumberOfSections','CreationYear'],axis=1)

le =  LabelEncoder()
for i in df:
    if df[i].dtype=='object':
        df[i] = le.fit_transform(df[i])
    else:
        continue

X= df.drop(['class'],axis=1)
y = df['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4 , random_state=40)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score

model = GaussianNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(y_pred[:50])

from sklearn.metrics import  accuracy_score, classification_report

accuracy = accuracy_score(y_pred, y_test)

print(accuracy)

print(classification_report(y_test,y_pred))
print('F1 Score: ',f1_score(y_test,y_pred,zero_division=1))

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
