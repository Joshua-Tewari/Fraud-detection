import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset to a Pandas DataFrame
try:
    credit_card_data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("The file 'creditcard.csv' was not found.")
    exit()

# Display basic information about the dataset
print("First 5 rows of the dataset:")
print(credit_card_data.head())
print("\nLast 5 rows of the dataset:")
print(credit_card_data.tail())
print("\nDataset information:")
credit_card_data.info()
print("\nNumber of missing values in each column:")
print(credit_card_data.isnull().sum())
print("\nDistribution of legit transactions and fraudulent transactions:")
print(credit_card_data['Class'].value_counts())

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print("\nShape of legit transactions:", legit.shape)
print("Shape of fraudulent transactions:", fraud.shape)

print("\nStatistical measures for legit transactions:")
print(legit.Amount.describe())
print("\nStatistical measures for fraudulent transactions:")
print(fraud.Amount.describe())

print("\nMean values per class:")
print(credit_card_data.groupby('Class').mean())

legit_sample = legit.sample(n=len(fraud), random_state=1)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print("\nNew dataset class distribution:")
print(new_dataset['Class'].value_counts())
print("\nMean values in the new dataset per class:")
print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nShapes of the datasets:")
print("X:", X.shape, "X_train:", X_train.shape, "X_test:", X_test.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('\nAccuracy on Training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on Test Data:', test_data_accuracy)

precision = precision_score(Y_test, X_test_prediction)
recall = recall_score(Y_test, X_test_prediction)
f1 = f1_score(Y_test, X_test_prediction)
print('\nPrecision score on Test Data:', precision)
print('Recall score on Test Data:', recall)
print('F1 score on Test Data:', f1)
