import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.callbacks import EarlyStopping
# ------------------------------------------------------ a -------------------------------------------------------------

# loading data
data_dict = pickle.load(open("california-housing-dataset.pkl", "rb"))

x_train, y_train = data_dict["x_train"], data_dict["y_train"]
x_test, y_test = data_dict["x_test"], data_dict["y_test"]

# Convert the data to a Pandas DataFrame for easier analysis and visualization
columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
df = pd.DataFrame(x_train, columns=columns)
df['Target_price'] = y_train  # target values

# Construct a validation set with 20% test and 80% train
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=40)

# print main info about the investigate feature distributions
df_info = df.describe()
#print(df_info)

# Visualize feature distributions
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    df[column].hist(bins=40)
    plt.title(column)
plt.show()

# Normalize the data since it is not
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# # Verify that the data is now normalized
# df_normalized = pd.DataFrame(x_train, columns=columns)
# df_normalized['Target_price'] = y_train
# df_info_normalized = df_normalized.describe()
# print(df_info_normalized)

# ------------------------------------------------------ b -------------------------------------------------------------

# neural network model
def create_NNM(hidden_units, hidden_layers):
    # Configure the model layers
    model = Sequential()

    # Input layer
    model.add(Input(shape=(8,)))

    # Hidden layers
    for _ in range(hidden_layers):
        model.add(Dense(hidden_units, activation='relu'))

    # Only one output layer since we have a regression task
    model.add(Dense(1, activation='linear'))
    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


# Define a function to train the model and return training and validation errors
def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs):
    # Configure the model training procedure
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                        verbose=0)

    # Calculate training and validation set errors
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    return train_error, val_error


# Define hyperparameters and architectures to test
hidden_units_list = [32, 64, 128]
hidden_layers_list = [1, 2, 3]
batch_size = 64
epochs = 50

# Create a table to store results
results = []

# Loop through different architectures and evaluate
for hidden_units in hidden_units_list:
    for hidden_layers in hidden_layers_list:
        model = create_NNM(hidden_units, hidden_layers)
        train_error, val_error = train_and_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs)
        results.append([hidden_units, hidden_layers, train_error, val_error])

# Print the results in a table
print("Architecture | Hidden Units | Hidden Layers | Train Error | Validation Error")
for result in results:
    print(
        f"{result[0]}-{result[1]}         | {result[0]}            | {result[1]}             | {result[2]:.4f}      | {result[3]:.4f}")


# ------------------------------------------------------ c -------------------------------------------------------------
