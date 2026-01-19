import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

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
df["Mother"] = ((df['Sex'] == 1) & (df['Parch'] > 0) & (df['Age'] > 18)).astype(int)

df["FareGroup"] = pd.qcut(df["Fare"], 4, labels=['Low', 'Medium', 'High', 'Very High'])
from sklearn.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler()
df["FareGroup"] = scaler.fit_transform(df[['Fare']])

df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# numerical_features = []
# categorical_features = []
# binary_features = []
#
# for col in df.columns:
#     if df[col].dtype in ['int64', 'float64']:
#         if df[col].nunique() == 2:
#             binary_features.append(col)
#         else:
#             numerical_features.append(col)
#     elif df[col].dtype == 'object':
#         categorical_features.append(col)

categorical_features = ['Embarked', 'Title', 'AgeGroup']

competed_df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

for column in competed_df.columns:
    if competed_df[column].dtype == 'bool':
        competed_df[column] = competed_df[column].map(int)

with open('cleared_titanic.csv', 'w') as f:
    f.write(competed_df.to_csv())