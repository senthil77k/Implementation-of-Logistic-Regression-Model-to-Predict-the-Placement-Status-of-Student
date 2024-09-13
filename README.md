# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.


## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: senthil kumaran c
RegisterNumber: 212223220103

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![Image-1 (1)](https://github.com/user-attachments/assets/9fee0b97-f5e8-4d37-9da2-2222672cbe0b)


### Checking the null() function
![Image-2](https://github.com/user-attachments/assets/92533e96-8089-42af-96c2-aabaa30af3a5)


### Print Data:
![Image-3](https://github.com/user-attachments/assets/8581de1e-046e-492a-9d51-89a2d5fd060f)


### Y_prediction array
![Image-4](https://github.com/user-attachments/assets/03470fe9-8bf0-4266-bbed-071541503378)


### Accuracy value
![Image-5](https://github.com/user-attachments/assets/f5589b0f-a44c-428e-a4d5-6cd9d6097888)



### Confusion array
![Image-6](https://github.com/user-attachments/assets/32ac9917-5608-4b6a-bef0-f31dd036b9ce)


### Classification Report
![Image-7](https://github.com/user-attachments/assets/e2439d40-1c49-491d-a96f-725bd5237091)


### Prediction of LR
![Image-8](https://github.com/user-attachments/assets/db7b0050-833c-457b-929f-058259fc19e8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
