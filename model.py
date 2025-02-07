import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

#import cleaned data
cleaned_house_data = pd.read_csv('dataset/cleaned_house_data.csv')

#split data into features and target
features = cleaned_house_data.drop(columns=['price'])
target = cleaned_house_data['price']

# Normalize target
target_mean = target.mean()
target_std = target.std()
target = (target - target_mean) / target_std  # Standardize target

#standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

#split data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#build model
model = tf.keras.Sequential ([tf.keras.layers.Dense(18, activation='relu', input_shape=(X_train.shape[1],))])
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# Define optimizer with a custom learning rate
learning_rate = 0.0008  # Adjust this value as needed
opt = Adam(learning_rate=learning_rate)

#compile the model
model.compile(optimizer= opt , loss='mean_absolute_error')

#Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2)

# validate the model
model.evaluate(X_test, y_test)

# Reverse the target standardization for better interpretability
predictions = model.predict(X_test)
predictions = (predictions * target_std) + target_mean  

# De-normalize actual prices for better interpretability
y_test = (y_test * target_std) + target_mean


#plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

#plot the actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Ideal diagonal line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()

# calculate R2 score
r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.4f}")


#save the model
model.save('house_price_model.keras')  # Recommended modern format








