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

df_median = df['Age'].median()
df["Age"] = df["Age"].fillna(df_median)
# print(df["Age"].isnull().sum())

df = df.drop(columns=['Cabin'])


