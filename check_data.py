import pandas as pd
import os

DATA_PATH = 'backend/balanced_urls.csv'
try:
    df = pd.read_csv(DATA_PATH, nrows=1000)
    print(df['result'].value_counts())
except Exception as e:
    print(e)
