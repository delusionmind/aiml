import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from six import StringIO


data = pd.read_csv("play_tennis.csv")
print("First 5 values of trained data is \n",data.head())
x = data.iloc[:,:-1]
print("First 5 values of trained data is \n",x.head())
y = data.iloc[:,-1]
print("\n First 5 values of Train output is \n", y.head())

le_outlook = LabelEncoder()
x.outlook = le_outlook.fit_transform(x.outlook)

le_temp = LabelEncoder()
x.temp = le_temp.fit_transform(x.temp)

le_humidity = LabelEncoder()
x.humidity = le_humidity.fit_transform(x.humidity)

le_windy = LabelEncoder()
x.windy = le_windy.fit_transform(x.wind)

print("\n Now the train data is \n", x.head())
le_play = LabelEncoder()
y = le_play.fit_transform(y)

print("\n Now the train data is \n", y)
