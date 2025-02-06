import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import the dataset
house_data= pd.read_csv('dataset/kc_house_data.csv')

#learn about the dataset
print(house_data.head(10))
print(house_data.describe())
print(house_data.info())

# Plot correlation heatmap
correlation_matrix = house_data.corr()  

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation with Price")
plt.show()
