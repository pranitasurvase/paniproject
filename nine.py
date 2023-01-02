import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer

#Loading the dataset
data = load_breast_cancer()

#Storing the dataset in pandas dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

#Adding the target variable
df['target'] = data.target

#Defining the independent and dependent variables
X = df.drop(['target'], axis=1)
y = df['target']

#Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#Building the SVM model
svc = SVC()
svc.fit(X_train,y_train)

#Predicting the values for test set
y_pred = svc.predict(X_test)

#Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

#Calculating the precision of the model
precision = precision_score(y_test, y_pred)

#Calculating the recall of the model
recall = recall_score(y_test, y_pred)

#Printing the accuracy, precision and recall of the model
print("Accuracy:", accuracy)
print("Precision:", precision) 
print("Recall:", recall)
