import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

sns.set_style("whitegrid")

fig, ax = plt.subplots(3, 3, figsize=(15 ,8))

fig.suptitle('Titanic analyze', fontsize=14)

ax[0, 0].hist(df["Survived"], bins=50)
ax[0, 0].set_xlabel('Survived')
ax[0, 0].set_ylabel('Count')
ax[0, 0].axvline(df['Survived'].mean(), color='r', label='Mean Survival', linestyle='--')
ax[0, 0].axvline(df['Survived'].mode()[0], color='orange', label='Mode Survival', linestyle='-')
ax[0, 0].axvline(df['Survived'].median(), color='red', label='Median Survival', linestyle=':')
ax[0, 0].legend()

ax[0, 1].hist(df["Fare"])
ax[0, 1].set_xlim(-50, 200)

sns.boxplot(data=df, x="Pclass", y="Age", ax=ax[0, 2])

survival_by_sex = df.groupby("Sex")["Survived"].mean().reset_index()
# print(survival_by_sex)
ax[1, 0].bar(["Male (0)", "Female (1)"], survival_by_sex["Survived"])

for i, v in enumerate(survival_by_sex["Survived"]):
    ax[1, 0].text(i, v+0.01, f"{v:.2f}", ha="center")

plt.show()

