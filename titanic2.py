import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

print(df['Age'].describe())

plt.figure(figsize=(10, 6))