import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# print(df['Age'].describe())

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Age'].dropna(), bins=50, kde=True)

plt.subplot(1, 2, 2)
sns.boxplot(df['Age'].dropna())

# plt.show()

df_age_median = df['Age'].median()
df["Age"] = df["Age"].fillna(df_age_median)
# print(df["Age"].isnull().sum())

df = df.drop(columns=['Cabin'])

# print(df['Embarked'].value_counts())
df_embarked_median = df['Embarked'].mode()[0]
# print(df_embarked_median)

df['Embarked'] = df['Embarked'].fillna(df_embarked_median)
# print(df.isnull().sum())

# print(df["Sex"].value_counts())
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)



