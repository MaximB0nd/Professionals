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

df["Title"] = df["Name"].str.extract("([a-zA-Z]+)\.", expand=False)

# print(df["Title"].value_counts())

mapper = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Mlle": "Rare",
    "Major": "Rare",
    "Col": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Ms": "Rare",
    "Sir": "Rare",
    "Lady": "Rare",
    "Mme": "Rare",
    "Don": "Rare",
    "Jonkheer": "Rare",
}

df["Title"] = df["Title"].map(mapper)
# print(df["Title"].value_counts())

def create_age_groups(age):
    if age <= 16:
        return 'Child'
    elif age <= 32:
        return 'Young Adult'
    elif age <= 48:
        return 'Adult'
    elif age <= 64:
        return 'Middle Aged'
    else:
        return 'Senior'

df["AgeGroup"] = df["Age"].map(create_age_groups)
# print(df["AgeGroup"].value_counts())

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df["isAlone"] = (df['FamilySize'] == 1).astype(int)
df["isLargeFamily"] = (df['FamilySize'] > 4).astype(int)
df['AgeClass'] = df['Age'] * df['Pclass']



