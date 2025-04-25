# EX:5 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (pandas, LabelEncoder, train_test_split, etc.).
2. Load the dataset using pd.read_csv().
3. Create a copy of the dataset and drop unnecessary columns (sl_no, salary).
4. Check for missing and duplicate values using isnull().sum() and duplicated().sum().
5. Encode categorical variables using LabelEncoder() to convert them into numerical values. 

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
### Developed by: SHAHIN J
### RegisterNumber: 212223040190 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Placement_Data.csv')
data.info()
```
![image](https://github.com/user-attachments/assets/1b3b39f6-1141-4b4f-af2c-840472668879)
```
data=data.drop(['sl_no'],axis=1)  # should run only once
data
```
![image](https://github.com/user-attachments/assets/96d623b9-4bf9-475a-b765-4f6158dd6e3c)
```
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes
```
![image](https://github.com/user-attachments/assets/4a646561-c08f-4e7d-8e3c-31c6d585309d)
```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data=data.drop(['salary'],axis=1)
data
```
![image](https://github.com/user-attachments/assets/9e00a82e-fc14-4a8a-9708-6af2d79aeff7)
```
x=data.iloc[:,:-1].values # :-1 means from starting till -1 (rows)
y=data.iloc[:,-1].values  # -1 means only the last column
x
```
![image](https://github.com/user-attachments/assets/18f50729-ef59-4d56-949f-8c54576d1d46)
```
y
```
![image](https://github.com/user-attachments/assets/a3b56579-bdca-483d-bf2a-a1bf9e448cbd)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
```
![image](https://github.com/user-attachments/assets/2808d3f4-3b49-497b-8af3-831012308a99)
```
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
clf.predict(x_test)
```
![image](https://github.com/user-attachments/assets/ad872406-3b69-4e98-8d28-bbc56691857c)
```
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test,clf.predict(x_test))
acc
```
![image](https://github.com/user-attachments/assets/1a8ddfe5-68c6-484a-ba40-eeedc6efa3be)
```
from sklearn.metrics import confusion_matrix
confusion = (y_test, clf.predict(x_test))
confusion
```
![image](https://github.com/user-attachments/assets/594be4ad-f704-44d7-94ab-e93290ef06a3)
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,clf.predict(x_test) )
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
![image](https://github.com/user-attachments/assets/2c0a6e96-5f89-4e67-bd4e-fcd9ff10efad)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
