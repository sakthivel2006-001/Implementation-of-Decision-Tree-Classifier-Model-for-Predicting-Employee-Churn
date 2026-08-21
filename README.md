# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: SAKTHIVEL S 

RegisterNumber:212223220090

```python
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```
## Output:
![{EB0C71C5-A916-4111-B19A-6C410986AC1D}](https://github.com/user-attachments/assets/7ca57902-b2a2-4ff7-863f-7c1f79b7db9a)


![{DFCF2198-E781-4526-A7B5-FF1A7EB3A2FA}](https://github.com/user-attachments/assets/a048a04b-2ad6-4620-a497-051254eb857a)


![{7EEDF7BD-57CE-49C5-898D-5761436F4A35}](https://github.com/user-attachments/assets/11fd07a2-524a-4458-b432-1f55c246e2b3)


![{5941F608-E807-48A2-A36D-1BE45DDFCBF1}](https://github.com/user-attachments/assets/4761a29e-5ddc-4e0e-b2f9-5d17e9e20b47)

![{E15987F7-1861-4124-A6F5-3AA593247995}](https://github.com/user-attachments/assets/1af4e6aa-f3b1-4443-9f51-e7328765d2ce)

![{79916BA2-F21E-4397-9D67-D2E2735561D3}](https://github.com/user-attachments/assets/375e3eab-1628-4054-8ad1-f13893da0880)

![{3A7E7758-0076-420C-A7F3-1B23AB6ACDD9}](https://github.com/user-attachments/assets/f2829566-0121-4c0c-8c0e-63e5a5a4d37b)

![{48880982-1AE6-47C7-A03E-6E1F89164ED8}](https://github.com/user-attachments/assets/c1fedf2d-2bf1-4ce9-860c-f592bba38b5c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
