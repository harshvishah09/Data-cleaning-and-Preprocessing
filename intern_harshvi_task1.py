# =====================================
# Data Cleaning & Preprocessing
# Titanic Dataset
# =====================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Step 2: Load Dataset
df = pd.read_csv('Titanic-Dataset.csv')


# Step 3: Explore Dataset
print("----- FIRST 5 ROWS -----")
print(df.head())

print("\n----- DATASET INFO -----")
print(df.info())

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())


# Step 4: Handle Missing Values

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column
df.drop(columns=['Cabin'], inplace=True)

print("\n----- AFTER CLEANING MISSING VALUES -----")
print(df.isnull().sum())


# Step 5: Convert Categorical to Numerical

# Convert Sex (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\n----- AFTER ENCODING -----")
print(df.head())


# Step 6: Normalize Data (Using StandardScaler)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\n----- AFTER NORMALIZATION -----")
print(df[['Age', 'Fare']].head())


# Step 7: Boxplot (Before Outlier Removal)

plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare Before Outlier Removal")
plt.show()


# Step 8: Remove Outliers (IQR Method)

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= Q1 - 1.5 * IQR) &
        (df['Fare'] <= Q3 + 1.5 * IQR)]


# Step 9: Boxplot (After Outlier Removal)

plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare After Outlier Removal")
plt.show()


# Step 10: Final Dataset Shape

print("\n----- FINAL DATASET SHAPE -----")
print(df.shape)


# Step 11: Save Cleaned Dataset

df.to_csv('cleaned_titanic.csv', index=False)

print("\n✅ Data Cleaning Completed Successfully!")