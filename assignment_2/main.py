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
#print(df_info)

# Visualize feature distributions
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)
    return model, model_name


# Define a function to train the model and return training and validation errors
def train_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, model_name, x_test, y_test):
    # Configure the model training procedure
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                        verbose=0)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)



    # plot results
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Evolution of training and validation error for model {}'.format(model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

    return test_loss, test_accuracy

# Define hyperparameters and architectures to test
hidden_units_list = [64, 128, 256]
hidden_layers_list = [1, 2, 3]
batch_size = 32
epochs = 70
learning_rate = 0.01

# optimizers
sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False)
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create a table to store results
results = []

# Loop through different architectures and evaluate
for hidden_units in hidden_units_list:
    for hidden_layers in hidden_layers_list:
        model, model_name = create_NNM(hidden_units, hidden_layers)
        test_loss, test_accuracy = train_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs,
                                                      model_name, x_test, y_test)

        results.append([hidden_units, hidden_layers, test_loss, test_accuracy])

# Print the results in a table
print("Architecture | Hidden Units | Hidden Layers | Train Error | Validation Error")
for result in results:
    print(
        f"{result[0]}-{result[1]}    | {result[0]}     | {result[1]}     | {result[2]:.4f}      | {result[3]:.4f}")


# ------------------------------------------------------ c -------------------------------------------------------------






