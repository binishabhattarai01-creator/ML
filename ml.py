import sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data= load_iris()
data

x=data.data

y=data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#training models

model = RandomForestClassifier()
model.fit(x_train, y_train) 

#Test the model
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

#Input and test model
model = RandomForestClassifier()
model.fit(x_train, y_train)

sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
print(prediction)
print(f"Predicted class: {data.target_names[prediction[0]]}")