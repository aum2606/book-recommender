import pandas as pd
import numpy as np

# Read first few rows of the data
df = pd.read_csv('data/data.csv')

print("\nDataset Info:")
print(df.info())

print("\nSample of ratings:")
print(df[['title', 'rating']].head())

print("\nRating statistics:")
print(df['rating'].describe())
