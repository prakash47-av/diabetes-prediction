import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

from sklearn.linear_model import LogisticRegression

df=pd.read_csv('C:/Users\pradeepa.s\OneDrive\Documents\diabetesprediction.csv')

df.info()
df.dtypes
df.duplicated()
df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()


df.isna().sum()


df.shape

df.describe()

df.value_counts()

df['smoking_history'].value_counts()

df['heart_disease'].value_counts()
df['gender'].value_counts()

encoder=LabelEncoder()

df['gender']=encoder.fit_transform(df['gender'])

df['smoking_history']=encoder.fit_transform(df['smoking_history'])
df.head()


X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)
X_train.shape,X_test.shape,X.shape



lr=LogisticRegression(max_iter=3000)

lr.fit(X_train,y_train)


y_predection=lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_predection)

lr_conf_matrix = confusion_matrix(y_test, y_predection)

lr_classification_rep = classification_report(y_test, y_predection)




print(f'lr_Accuracy: {lr_accuracy:.2f}')

print('\nlr_Confusion Matrix:')

print(lr_conf_matrix)

print('\nlr_Classification Report:')

print(lr_classification_rep)

print("Training Score:",lr.score(X_train,y_train)*100,'%')

print("Testing Score:",lr.score(X_test,y_test)*100,'%')




# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predection)
print(cm)



acc=(sum(np.diag(cm))/len(y_test))
acc


# Displaying the predictions with index values
predictions_with_index = pd.DataFrame({
    'Index': X_test.index,  # Getting the index of the test data
    'Predicted': y_predection
})


