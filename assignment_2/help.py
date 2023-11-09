import numpy as np
import pickle
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.optimizers.legacy import SGD as momSGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, elu, sigmoid, tanh
from sklearn.metrics import mean_squared_error


# Create a function to build the model with a variable number of hidden layers
def build_model_adam(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = 32
    l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    activation = tanh
    num_neur = 128

    # Number of hidden layers (variable between 1 and 4 layers)
    num_hidden_layers = 2

    model = Sequential()
    model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
                    kernel_regularizer=l2(l2_regularization)))
    model.add(BatchNormalization())

    for _ in range(num_hidden_layers):
        model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=1))

    lr_schedule = schedules.ExponentialDecay(
        learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


# Define the objective function for Optuna
def objective_adam(trial):
    model = build_model_adam(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x=x_train, y=y_train, batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]), epochs=100,
              validation_data=(x_val, y_val),
              shuffle=True, verbose=0, callbacks=[early_stopping])

    y_pred = model.predict(scaled_x_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse


# Create a function to build the model for SGD
def build_model_SGD(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True)
    batch_size = 32
    l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    activation = tanh
    num_neur = 128

    # Number of hidden layers (variable between 1 and 4 layers)
    num_hidden_layers = 2

    model = Sequential()
    model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
                    kernel_regularizer=l2(l2_regularization)))
    model.add(BatchNormalization())

    for _ in range(num_hidden_layers):
        model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=1))

    lr_schedule = schedules.ExponentialDecay(
        learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

    model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


# Define the objective function for SGD
def objective_SGD(trial):
    model = build_model_SGD(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True)

    model.fit(x=x_train, y=y_train, batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]), epochs=100,
              validation_data=(x_val, y_val),
              shuffle=True, verbose=0, callbacks=[early_stopping])

    y_pred = model.predict(scaled_x_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse


# Create a function to build the model for Momentum SGD
def build_model_momSGD(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True)
    batch_size = 32
    l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    activation = tanh
    num_neur = 128

    # Number of hidden layers (variable between 1 and 4 layers)
    num_hidden_layers = 2

    model = Sequential()
    model.add(Dense(units=num_neur, input_shape=(x_train.shape[1],), activation=activation,
                    kernel_regularizer=l2(l2_regularization)))
    model.add(BatchNormalization())

    for _ in range(num_hidden_layers):
        model.add(Dense(units=num_neur, activation=activation, kernel_regularizer=l2(l2_regularization)))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=1))

    lr_schedule = schedules.ExponentialDecay(
        learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

    model.compile(optimizer=momSGD(learning_rate=lr_schedule), loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model


# Define the objective function for Momentum SGD
def objective_mom_SGD(trial):
    model = build_model_momSGD(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(x=x_train, y=y_train, batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]), epochs=100,
              validation_data=(x_val, y_val),
              shuffle=True, verbose=0, callbacks=[early_stopping])

    y_pred = model.predict(scaled_x_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse


# Load data
data_dict = pickle.load(open('california-housing-dataset.pkl', 'rb'))
x_train, y_train = data_dict['x_train'], data_dict['y_train']
x_test, y_test = data_dict['x_test'], data_dict['y_test']

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(scaled_x_train, y_train, test_size=0.2, random_state=42)

num_trials = 20

print("ADAM")
study = optuna.create_study(direction="minimize")
study.optimize(objective_adam, n_trials=num_trials)
best_params = study.best_params
best_mse = study.best_value
print("Best Hyperparameters for Adam:", best_params)
print("Best MSE for Adam:", best_mse)

print("========================================================================\n\n\n")

print("SGD")
study = optuna.create_study(direction="minimize")
study.optimize(objective_SGD, n_trials=num_trials)
best_params = study.best_params
best_mse = study.best_value
print("Best Hyperparameters for SGD:", best_params)
print("Best MSE for SGD:", best_mse)

print("========================================================================\n\n\n")

print("Momentum SGD")
study = optuna.create_study(direction="minimize")
study.optimize(objective_mom_SGD, n_trials=num_trials)
best_params = study.best_params
best_mse = study.best_value
print("Best Hyperparameters for Momentum SGD:", best_params)
print("Best MSE for Momentum SGD:", best_mse)
