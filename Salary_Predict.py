from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

datas = {
    'age': [10, 20, 30, 40, 50, 60, 70, 80, 90],
    'salary': [21000, 41000, 61000, 81000, 110000, 121000, 141000, 161000, 181000]
}
df2 = pd.DataFrame(datas)

# df2[['age','salary']].boxplot(vert = False)


Q_1 = np.percentile( df2 [ 'age' ] , 25)

# print(Q_1)

Q_3 = np.percentile(df2 [ 'age' ] , 75)

# print(Q_3)

IQRs = Q_3 - Q_1 

# print(IQRs)

for col in ['age' , 'salary']:

    Q_1 = np.percentile( df2 [ col ] , 25)

    # print(Q_1)

    Q_3 = np.percentile(df2 [ col ] , 75)

    # print(Q_3)

    IQRs = Q_3 - Q_1 
    
    lower = Q_1 - 1.5 * IQRs

    upper  = Q_3 + 1.5 *IQRs

    df2 = df2 [ (df2[col] > lower) & (df2[col] < upper)] 
    
# print(df2)


scaler = MinMaxScaler()

df2_scaled = pd.DataFrame(scaler.fit_transform(df2) , columns = df2.columns)

# print(df2_scaled.round(3))



X = df2[['age']]

Y = df2['salary']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train , Y_train)

Y_pred = model.predict(X_test)

Y_pred

user_age = float(input("Enter your age to predict salary: "))

predicted_salary = model.predict([[user_age]])

print("Predicted Salary for age", user_age, "is:", round(predicted_salary[0], 2))
