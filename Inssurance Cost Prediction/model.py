# importing libraries :

import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

#importing datasets  
df= pd.read_csv(r'C:\Users\PC\Desktop\Project\Insurance Cost Predection\insurance.csv')

#Extraction variables:
x=df.iloc[:,:-1].values
y=df.iloc[:,6].values
 
#Encodimg Catgorical data:  
encoded_df = pd.get_dummies(df, columns=['sex','smoker','region'], dtype=int)

#Defeine variables:
X = encoded_df.drop('expenses', axis=1)
y = encoded_df['expenses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients and intercept:
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Predict on the test set
y_pred = model.predict(X_test)

#evaluate the model's performance using metrics like Mean Squared Error or R-squared

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Save the model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)