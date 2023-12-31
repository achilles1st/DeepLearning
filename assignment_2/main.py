import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.optimizers.legacy import SGD as momSGD
from tensorflow.keras.optimizers import schedules

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.callbacks import EarlyStopping


# ---------------------------------------------------Functions----------------------------------------------------------
# ------------------------------------------------------ b -------------------------------------------------------------
# neural network model


def create_NNM_regression(hidden_units, hidden_layers):
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
    model.summary()

    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])

    return model, model_name


def train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val, batch_size, epochs, x_test, y_test):
    # Configure the model training procedure
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                        verbose=0, callbacks=[early_stopping])

    train_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

    # # plot results
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Evolution of training and validation error for model {}'.format(model_name) + f"_batch_{batch_size}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
    name = (str(model_name) + '-' + str(batch_size))
    plt.savefig(f'Final.png')

    val_error = history.history['val_loss']
    train_loss = history.history['loss']

    return train_loss[-1], test_accuracy, val_error[-1]


# ------------------------------------------------------ c -------------------------------------------------------------


def create_NNM_regression_ADAM(hidden_units, hidden_layers):
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
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse'])

    return model, model_name


def create_NNM_regression_SGD(hidden_units, hidden_layers):
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
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['mse'])

    return model, model_name


def create_NNM_regression_momSGD(hidden_units, hidden_layers):
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
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=momSGD(), metrics=['mse'])

    return model, model_name


def create_NNM_regression_learning_rate(hidden_units, hidden_layers, learning_rate):
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
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=['mse'])

    return model, model_name


# ------------------------------------------------------ d -------------------------------------------------------------


def train_evaluate_model_whole_training_set(model, x_train, y_train, batch_size, epochs, model_name, x_test, y_test):
    # Configure the model training procedure

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=30, restore_best_weights=True)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=0, callbacks=[early_stopping])

    train_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred = model.predict(x_test)

    # # # plot results
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Evolution of training error for model {}'.format(model_name) + f"_batch_{batch_size}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
    plt.savefig(f'figures/Whole_training.png')

    x = np.polyfit(y_test, y_pred, 1)
    z = np.array([x[0][0], x[1][0]])
    p = np.poly1d(z)
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, p(y_test), color='red')
    plt.xlabel('test values')
    plt.ylabel('pred values')
    plt.title('Scatter Plot predicted vs. test')
    plt.show()
    plt.savefig('figures/Scatterplot_predicted_test_trendline.png')

    y = np.arange(0, len(y_pred), 1)

    plt.figure()
    plt.scatter(y, y_pred)
    plt.scatter(y, y_test)
    plt.legend()
    plt.ylabel('values')
    plt.title('Scatter Plot predicted vs. test')
    plt.show()
    plt.savefig('figures/Scatterplot_predicted_test.png')

    loss = history.history['loss']
    return train_loss, test_accuracy, loss[-1]




# ------------------------------------------------------ e -------------------------------------------------------------


def create_NNM_classification_classification(hidden_units, hidden_layers):
    model_name = "hl_{}_hu_{}".format(hidden_layers, hidden_units)

    # Configure the model layers
    model = Sequential()

    # Input layer
    model.add(Dense(units=hidden_units, input_dim=8, kernel_initializer='normal', activation='tanh'))

    # Hidden layers
    for _ in range(hidden_layers):
        model.add(Dense(hidden_units, activation='tanh'))

    # Only one output layer since we have a regression task
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, model_name


def train_evaluate_model_classification(model, x_train, y_train, batch_size, epochs, model_name, x_test, y_test):
    # Configure the model training procedure
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=0, callbacks=[early_stopping])

    train_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

    # # # plot results
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title(
        'Binary evolution of training and validation error for model {}'.format(model_name) + f"_batch_{batch_size}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
    name = (str(model_name) + '-' + str(batch_size))
    plt.savefig(f'Binary.png')

    return train_loss, test_accuracy


# ------------------------------------------------------Init------------------------------------------------------------
# loading data
data_dict = pickle.load(open("california-housing-dataset.pkl", "rb"))

x_train, y_train = data_dict["x_train"], data_dict["y_train"]
x_train_full, y_train_full = data_dict["x_train"], data_dict["y_train"]
x_test, y_test = data_dict["x_test"], data_dict["y_test"]

# Convert the data to a Pandas DataFrame for easier analysis and visualization
columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
df = pd.DataFrame(x_train, columns=columns)
df['Target_price'] = y_train  # target values

# Normalize the data since it is not
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train_full = scaler.fit_transform(x_train_full)

# Construct a validation set with 25% test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# final used architecture
hidden_units = 32
hidden_layers = 2
batch_size = 128
epochs = 300
learning_rate = 0.01

# ------------------------------------------------------Mains------------------------------------------------------------
# ------------------------------------------------------ a -------------------------------------------------------------


# print main info about the investigate feature distributions
# df_info = df.describe()

# print(df_info)

# Visualize feature distributions
# plt.figure(figsize=(12, 8))
# for i, column in enumerate(df.columns):
#     plt.subplot(3, 3, i + 1)
#     df[column].hist(bins=40)
#     plt.title(column)
# plt.show()

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print("------------------")
#
# print(max(y_train))
# print(min(y_train))
# print("------------------")
# med_income = x_train[:, 0]
# print(min(med_income))
# print(max(med_income))
# house_age = x_train[:, 1]
# print(min(house_age))
# print(max(house_age))
# avg_rooms = x_train[:, 2]
# print(min(avg_rooms))
# print(max(avg_rooms))
# avg_brooms = x_train[:, 3]
# print(min(avg_brooms))
# print(max(avg_brooms))
# pop = x_train[:, 4]
# print(min(pop))
# print(max(pop))
# avg_occup = x_train[:, 5]
# print(min(avg_occup))
# print(max(avg_occup))
# coordinates = x_train[:, -2:]
# target = y_train
#
#
#
# sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1],
#                 size=target, hue=target, palette="viridis", alpha=0.5)
# plt.legend(title="Value", loc="upper right")
# plt.axis([31, 43, -125, -113])
# plt.title("North-West position of houses")
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.show()

# ------------------------------------------------------ b -------------------------------------------------------------


#
# # Define hyperparameters and architectures to test
# # hidden_units_list = [4, 8, 16, 32, 64, 128, 256]
# # hidden_layers_list = [1, 2, 3]
# # batch_sizes = [8, 16, 32, 64, 128, 256]

# # final used architecture
# hidden_units_list = [32]
# hidden_layers_list = [2]
# batch_sizes = [128]
# epochs = 300
# #
# # # Create a table to store results
# results = []
# for batch_size in batch_sizes:
#     # Loop through different architectures and evaluate
#     for hidden_units in hidden_units_list:
#         for hidden_layers in hidden_layers_list:
#             model, model_name = create_NNM_regression(hidden_units, hidden_layers)
#             train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                            batch_size, epochs, x_test, y_test)
#
#
#             results.append([hidden_units, hidden_layers, train_loss, batch_size, val_error])
#
# # Print the results in a table
# print(" Hidden Units | Hidden Layers | batch size | Train Error | Val Error")
# for result in results:
#     print("{}|{}|{}|{}|{}".format(result[0], result[1], result[3], result[2], result[4]))

# ------------------------------------------------------ c -------------------------------------------------------------
#
# import numpy as np
# import pickle
# import optuna
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.experimental import SGD
# from tensorflow.keras.optimizers.legacy import SGD as momSGD
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import schedules
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.activations import relu, elu, sigmoid, tanh
# from sklearn.metrics import mean_squared_error
#
#
# # Create a function to build the model with a variable number of hidden layers
# def build_model_adam(trial):
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
#     batch_size = 128
#     l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
#     activation = tanh
#     num_neur = 32
#
#     # Number of hidden layers (variable between 1 and 4 layers)
#     num_hidden_layers = 2
#
#     model = Sequential()
#     model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
#                     kernel_regularizer=l2(l2_regularization)))
#     model.add(BatchNormalization())
#
#     for _ in range(num_hidden_layers):
#         model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(units=1))
#
#     lr_schedule = schedules.ExponentialDecay(
#         learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
#
#     model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error',
#                           metrics=['mean_absolute_error'])
#
#     return model
#
#
# # Define the objective function for Optuna
# def objective_adam(trial):
#     model = build_model_adam(trial)
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#
#     model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
#               validation_data=(x_val, y_val),
#               shuffle=True, verbose=0, callbacks=[early_stopping])
#
#     y_pred = model.predict(scaled_x_test)
#     mse = mean_squared_error(y_test, y_pred)
#
#     return mse
#
#
# # Create a function to build the model for SGD
# def build_model_SGD(trial):
#     learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, log=True)
#     batch_size = 128
#     l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
#     activation = tanh
#     num_neur = 32
#
#     # Number of hidden layers (variable between 1 and 4 layers)
#     num_hidden_layers = 2
#
#     model = Sequential()
#     model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
#                     kernel_regularizer=l2(l2_regularization)))
#     model.add(BatchNormalization())
#
#     for _ in range(num_hidden_layers):
#         model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(units=1))
#
#     lr_schedule = schedules.ExponentialDecay(
#         learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
#
#     model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='mean_squared_error', metrics=['mean_absolute_error'])
#
#     return model
#
#
# # Define the objective function for SGD
# def objective_SGD(trial):
#     model = build_model_SGD(trial)
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights = True)
#
#     model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
#               validation_data=(x_val, y_val),
#               shuffle=True, verbose=0, callbacks=[early_stopping])
#
#     y_pred = model.predict(scaled_x_test)
#     mse = mean_squared_error(y_test, y_pred)
#
#     return mse
#
#
# # Create a function to build the model for Momentum SGD
# def build_model_momSGD(trial):
#     learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, log=True)
#     batch_size = 128
#     l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
#     activation = tanh
#     num_neur = 32
#
#     # Number of hidden layers (variable between 1 and 4 layers)
#     num_hidden_layers = 2
#
#     model = Sequential()
#     model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
#                     kernel_regularizer=l2(l2_regularization)))
#     model.add(BatchNormalization())
#
#     for _ in range(num_hidden_layers):
#         model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(units=1))
#
#     lr_schedule = schedules.ExponentialDecay(
#         learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
#
#     model.compile(optimizer=momSGD(learning_rate=lr_schedule), loss='mean_squared_error',
#                   metrics=['mean_absolute_error'])
#
#     return model
#
#
# # Define the objective function for Momentum SGD
# def objective_mom_SGD(trial):
#     model = build_model_momSGD(trial)
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#
#     model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
#               validation_data=(x_val, y_val),
#               shuffle=True, verbose=0, callbacks=[early_stopping])
#
#     y_pred = model.predict(scaled_x_test)
#     mse = mean_squared_error(y_test, y_pred)
#
#     return mse
#
#
#
# num_trials = 100
#
# print("ADAM")
# study = optuna.create_study(direction="minimize")
# study.optimize(objective_adam, n_trials=num_trials)
# best_params = study.best_params
# best_mse = study.best_value
# print("Best Hyperparameters for Adam:", best_params)
# print("Best MSE for Adam:", best_mse)
#
# print("========================================================================\n\n\n")
#
# print("SGD")
# study = optuna.create_study(direction="minimize")
# study.optimize(objective_SGD, n_trials=num_trials)
# best_params = study.best_params
# best_mse = study.best_value
# print("Best Hyperparameters for SGD:", best_params)
# print("Best MSE for SGD:", best_mse)
#
# print("========================================================================\n\n\n")
#
# print("Momentum SGD")
# study = optuna.create_study(direction="minimize")
# study.optimize(objective_mom_SGD, n_trials=num_trials)
# best_params = study.best_params
# best_mse = study.best_value
# print("Best Hyperparameters for Momentum SGD:", best_params)
# print("Best MSE for Momentum SGD:", best_mse)

# ======================================================================================================================
# ======================================================================================================================

# Other Try


# model, model_name = create_NNM_regression_ADAM(hidden_units, hidden_layers)
# train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                     batch_size, epochs, x_test, y_test)
# print(" Classifier | Train Error |  Val Error")
# print("{}|{}|{}".format("Adam", train_loss, val_error))
#
# model, model_name = create_NNM_regression_SGD(hidden_units, hidden_layers)
# train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                     batch_size, epochs, x_test, y_test)
# print(" Classifier | Train Error |  Val Error")
# print("{}|{}|{}".format("SGD", train_loss, val_error))
#
# model, model_name = create_NNM_regression_momSGD(hidden_units, hidden_layers)
# train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                     batch_size, epochs, x_test, y_test)
# print(" Classifier | Train Error |  Val Error")
# print("{}|{}|{}".format("momSGD", train_loss, val_error))

#
# learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# results = []
#
# for learning_rate in learning_rates:
#     model, model_name = create_NNM_regression_learning_rate(hidden_units, hidden_layers, learning_rate)
#     train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                          batch_size, epochs, x_test, y_test)
#     results.append([learning_rate, "no", train_loss, val_error])
#
# for learning_rate in learning_rates:
#     lr_schedule = schedules.ExponentialDecay(
#         learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
#     model, model_name = create_NNM_regression_learning_rate(hidden_units, hidden_layers, lr_schedule)
#     train_loss, test_accuracy, val_error = train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
#                                                                          batch_size, epochs, x_test, y_test)
#     results.append([learning_rate, "yes", train_loss, val_error])
#
# print(" Learning Rate | Scheduled | Train Error |  Val Error")
# for result in results:
#     print("{}|{}|{}|{}".format(result[0], result[1], result[2], result[3]))

# ------------------------------------------------------ d -------------------------------------------------------------
# final used architecture
lr_schedule = schedules.ExponentialDecay(
    learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

model, model_name = create_NNM_regression_learning_rate(hidden_units, hidden_layers, lr_schedule)

train_evaluate_model_with_val(model, x_train, y_train, x_val, y_val,
                              batch_size, epochs, x_test, y_test)

train_loss, test_accuracy, loss = train_evaluate_model_whole_training_set(model, x_train_full, y_train_full,
                                                                          batch_size, epochs, model_name, x_test,
                                                                          y_test)

print("Whole training set")
print("Train Error")
print("{}".format(loss))

# ------------------------------------------------------ e -------------------------------------------------------------

# y_train[y_train < 2], y_test[y_test < 2] = 0, 0
# y_train[y_train >= 2], y_test[y_test >= 2] = 1, 1
#
# model, model_name = create_NNM_classification_classification(hidden_units, hidden_layers)
# train_loss, test_accuracy = train_evaluate_model_classification(model, x_train, y_train, batch_size, epochs, model_name,
#                                                                x_test, y_test)
# print("Test Loss | Test accuracy")
# print("{}|{}".format(train_loss, test_accuracy))
