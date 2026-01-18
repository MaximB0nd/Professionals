import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.info())
print(df.describe())
print(df.describe(include=['O']))
print(df.isnull().sum())
print(df.shape[0])
print(df.shape[1])
print(df.tail(4))
print(df['Survived'].unique())

print(df["Survived"].where(df["Survived"] == 1).isnull().sum() / df.shape[0] * 100)
print(df["Survived"].where(df["Survived"] == 0).isnull().sum() / df.shape[0] * 100)
print(df["Pclass"].mode()[0])

# PLT

fig, ax = plt.subplots(3, 3, figsize=(12, 8))

sns.countplot(data=df, x='Survived', ax=ax[0, 0])
ax[0, 0].set_title('Распределение Survived (0=погиб, 1=выжил)')

# 2. Распределение по классам
sns.countplot(data=df, x='Pclass', ax=ax[0, 1])
ax[0, 1].set_title('Распределение по классам кают')

# 3. Распределение по полу
sns.countplot(data=df, x='Sex', ax=ax[1, 0])
ax[1, 0].set_title('Распределение по полу')

# 4. Распределение портов посадки
sns.countplot(data=df, x='Embarked', ax=ax[1, 1])
ax[1, 1].set_title('Распределение по порту посадки')

sns.countplot(data=df, x='Age', ax=ax[2, 0])
ax[2, 0].set_title("Age")

plt.tight_layout()
plt.show()




