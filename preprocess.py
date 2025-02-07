import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#import the dataset
house_data= pd.read_csv('dataset/kc_house_data.csv')

#learn about the dataset
print(house_data.head(10))
print(house_data.describe())
print(house_data.info())

# Drop non-numeric columns before computing correlation
house_data_numeric = house_data.drop(columns=['date'])

# Compute correlation matrix only on numeric features
correlation_matrix = house_data_numeric.corr()

# Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation with Price")
plt.show()

 # Drop columns with correlation less than 0.1
house_data = house_data.drop(columns=['id', 'date'])

# Compute average price per zipcode
zipcode_target = house_data.groupby('zipcode')['price'].mean()

# Map the average price to each row
house_data['zipcode'] = house_data['zipcode'].map(zipcode_target)


# scale the data using StandardScaler so that all the features are on the same scale
scaler = StandardScaler()
features = ['sqft_living', 'sqft_above', 'sqft_basement', 'lat', 'long']
house_data[features] = scaler.fit_transform(house_data[features])

# print the first 5 rows of the cleaned dataset
print(house_data.head())

#save the cleaned dataset
house_data.to_csv('dataset/cleaned_house_data.csv', index=False)
