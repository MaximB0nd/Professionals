import requests
import pandas as pd

# Скачивание и сохранение файла
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
response = requests.get(url)

with open('titanic.csv', 'wb') as f:
    f.write(response.content)