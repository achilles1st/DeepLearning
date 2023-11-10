import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Normalize the data since it is not
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Construct a validation set with 25% test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
# print main info about the investigate feature distributions
df_info = df.describe()
# print(df_info)

# Visualize feature distributions
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    df[column].hist(bins=40)
    plt.title(column)
plt.show()


# ------------------------------------------------------ b -------------------------------------------------------------

# neural network model
def create_NNM(hidden_units, hidden_layers):
    model_name = "hl_{}_hu_{}".format(hidden_layers, hidden_units)

    # Configure the model layers
    model = Sequential()

    # Input layer
    model.add(Dense(units=hidden_units, input_dim=8, kernel_initializer='normal', activation='tanh'))

    # Hidden layers
    for _ in range(hidden_layers):
        model.add(Dense(hidden_units, activation='tanh'))

    # Only one output layer since we have a regression task
    model.add(Dense(1, activation='linear'))
    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    return model, model_name, early_stopping


# Define a function to train the model and return training and validation errors
def train_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, model_name, x_test, y_test,
                         early_stopping):
    # Configure the model training procedure
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                        verbose=0, callbacks=[early_stopping])

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

    # # # plot results
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Evolution of training and validation error for model {}'.format(model_name) + f"_batch_{batch_size}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    # plt.show()

    plt.savefig(f'figures/{model_name}' + f"_batch_{batch_size}" + ".png")

    val_error = history.history['val_loss']
    return test_loss, test_accuracy, val_error[-1]


# Define hyperparameters and architectures to test
hidden_units_list = [4, 8, 16, 32, 64, 128, 256]
hidden_layers_list = [1, 2, 3]
batch_sizes = [4, 8, 16, 32,64,128,256]
epochs = 300

# Create a table to store results
results = []
for batch_size in batch_sizes:
    # Loop through different architectures and evaluate
    for hidden_units in hidden_units_list:
        for hidden_layers in hidden_layers_list:
            model, model_name, early_stopping = create_NNM(hidden_units, hidden_layers)
            test_loss, test_accuracy, val_error = train_evaluate_model(model, x_train, y_train, x_val, y_val,
                                                                       batch_size, epochs, model_name, x_test, y_test,
                                                                       early_stopping)

            results.append([hidden_units, hidden_layers, test_loss, batch_size, val_error])
            print(hidden_layers)

# Print the results in a table
print(" Hidden Units | Hidden Layers | batch size | Train Error | Val Error")
for result in results:
    print("{}|{}|{}|{}|{}".format(result[0], result[1], result[3], result[2], result[4]))

# ------------------------------------------------------ c -------------------------------------------------------------
hidden_unit = 128
hidden_layer = 2
learning_rates = [0.01]

# optimizers
# sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False)
# adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
