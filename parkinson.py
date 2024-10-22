# -*- coding: utf-8 -*-
"""Parkinson.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ay0XeJqrSHuBpOUVKC_eBdrKqZg248Ad
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# separating the features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# define the models
svm_model = svm.SVC(kernel='linear', probability=True)  # SVM with probability estimates
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)  # Random Forest model

# hybrid model using voting classifier
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' voting averages probabilities

# training the hybrid model on the training data
hybrid_model.fit(X_train_pca, Y_train)

# accuracy score on training data
X_train_prediction = hybrid_model.predict(X_train_pca)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = hybrid_model.predict(X_test_pca)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# prediction on new data
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

# applying PCA to input data
input_data_pca = pca.transform(std_data)

# prediction using the hybrid model
prediction = hybrid_model.predict(input_data_pca)
print(prediction)

if (prediction[0] == 0):
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

# Configure Git with your GitHub email and name
!git config --global user.email "mjanish19@gmail.com"
!git config --global user.name "Anish MJ"

!apt-get install git

# Correct GitHub repository URL
!git clone https://github.com/anishmj/Parkinson_Project.git

!ls /content/Parkinson_Project

# Move your code file into the repo
!mv /content/your_parkinson_code.py /content/Parkinson_Project/

# Commented out IPython magic to ensure Python compatibility.
# Navigate to your repo folder
# %cd https://github.com/anishmj/Parkinson_Project.git

# Add the files to staging
!git add .

# Commit the changes
!git commit -m "Added Parkinson code"

# Push the changes to GitHub
!git push origin main

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# separating the features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# define the models
svm_model = svm.SVC(kernel='linear', probability=True)  # SVM with probability estimates
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)  # Random Forest model

# hybrid model using voting classifier
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' voting averages probabilities

# training the hybrid model on the training data
hybrid_model.fit(X_train_pca, Y_train)

# accuracy score on training data
X_train_prediction = hybrid_model.predict(X_train_pca)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = hybrid_model.predict(X_test_pca)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# prediction on new data
input_data = (0, 0, 0, 0, 0,0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

# applying PCA to input data
input_data_pca = pca.transform(std_data)

# prediction using the hybrid model
prediction = hybrid_model.predict(input_data_pca)
print(prediction)

if (prediction[0] == 0):
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# separating the features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# define the models
svm_model = svm.SVC(kernel='linear', probability=True)  # SVM with probability estimates
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)  # Random Forest model

# hybrid model using voting classifier
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' voting averages probabilities

# training the hybrid model on the training data
hybrid_model.fit(X_train_pca, Y_train)

# accuracy score on training data
X_train_prediction = hybrid_model.predict(X_train_pca)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = hybrid_model.predict(X_test_pca)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# prediction on new data
input_data = (152.31, 210.76, 90.24, 0.00891, 0.000058, 0.00412, 0.00455, 0.01237,
              0.03824, 0.320, 0.1745, 0.02312, 0.02987, 0.05234, 0.03567, 18.45,
              0.497, 0.753, -7.115, 0.269, 2.452, 0.189)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

# applying PCA to input data
input_data_pca = pca.transform(std_data)

# prediction using the hybrid model
prediction = hybrid_model.predict(input_data_pca)
print(prediction)

if (prediction[0] == 0):
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# separating the features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# define the models
svm_model = svm.SVC(kernel='linear', probability=True)  # SVM with probability estimates
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)  # Random Forest model

# hybrid model using voting classifier
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' voting averages probabilities

# training the hybrid model on the training data
hybrid_model.fit(X_train_pca, Y_train)

# accuracy score on training data
X_train_prediction = hybrid_model.predict(X_train_pca)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = hybrid_model.predict(X_test_pca)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# prediction on new data
input_data = (145.78,198.56,82.12,0.00945,0.000065,0.00458,0.00491,0.01374,
              0.04123,0.355,0.01897,0.02467,0.03123,0.05691,0.03845,17.89,
              0.521,.781,-7.452,.275,2.563,.201)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

# applying PCA to input data
input_data_pca = pca.transform(std_data)

# prediction using the hybrid model
prediction = hybrid_model.predict(input_data_pca)
print(prediction)

if (prediction[0] == 0):
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# separating the features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # You can adjust the number of components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# define the models
svm_model = svm.SVC(kernel='linear', probability=True)  # SVM with probability estimates
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)  # Random Forest model

# hybrid model using voting classifier
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('rf', rf_model)
], voting='soft')  # 'soft' voting averages probabilities

# training the hybrid model on the training data
hybrid_model.fit(X_train_pca, Y_train)

# accuracy score on training data
X_train_prediction = hybrid_model.predict(X_train_pca)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data: ', training_data_accuracy)

# accuracy score on test data
X_test_prediction = hybrid_model.predict(X_test_pca)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on test data: ', test_data_accuracy)

# prediction on new data
input_data = (210.23,239.67,190.45,0.00110,0.000007,0.00078,0.00090,0.00234,
              0.01476,0.125,0.00678,0.00845,0.01123,0.02034,0.0123,25.34,
              0.312,0.689,-4.120,0.175,2.067,0.078)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

# applying PCA to input data
input_data_pca = pca.transform(std_data)

# prediction using the hybrid model
prediction = hybrid_model.predict(input_data_pca)
print(prediction)

if (prediction[0] == 0):
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")

import pickle

# Train the hybrid model
hybrid_model.fit(X_train, Y_train)

# Save the trained model to a .sav file
model_filename = 'parkinsons_hybrid_model.sav'
pickle.dump(hybrid_model, open(model_filename, 'wb'))

print(f"Model saved as {model_filename}")

with open('scaler.sav', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)