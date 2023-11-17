import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.optimizers.experimental import SGD
from keras.optimizers.legacy import SGD as momSGD
from keras.optimizers import schedules

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.callbacks import EarlyStopping

import numpy as np
import pickle
import optuna
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.optimizers.experimental import SGD
from keras.optimizers.legacy import SGD as momSGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import schedules
from keras.regularizers import l2
from keras.activations import relu, elu, sigmoid, tanh
from sklearn.metrics import mean_squared_error


class HousingRegressionModel:
    def __init__(self, data_dict):
        self.x_train, self.y_train = data_dict["x_train"], data_dict["y_train"]
        self.x_train_full, self.y_train_full = data_dict["x_train"], data_dict["y_train"]
        self.x_test, self.y_test = data_dict["x_test"], data_dict["y_test"]
        self.preprocess_data()

    def preprocess_data(self):
        # Convert data to a Pandas DataFrame for easier analysis and visualization
        columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        df = pd.DataFrame(self.x_train_full, columns=columns)
        df['Target_price'] = self.y_train_full # target values

        # Normalize the data
        scaler = StandardScaler()
        self.x_train_full = scaler.fit_transform(self.x_train_full)
        self.x_test = scaler.transform(self.x_test)

        # Construct a validation set with 25% test
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train_full, self.y_train, test_size=0.25, random_state=42
        )

        # Visualize feature distributions
        plt.figure(figsize=(12, 8))
        for i, column in enumerate(df.columns):
            plt.subplot(3, 3, i + 1)
            df[column].hist(bins=40)
            plt.title(column)
        plt.show()

    def execute_part_a(self):
        df_info = pd.DataFrame(self.x_train, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
                                                      'AveOccup', 'Latitude', 'Longitude']).describe()
        print(df_info)
        # investigating data
        print(self.x_train_full.shape)
        print(self.y_train_full.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)
        print("------------------")

        print(max(self.y_train_full))
        print(min(self.y_train_full))
        print("------------------")
        med_income = self.x_train_full[:, 0]
        print(min(med_income))
        print(max(med_income))
        house_age = self.x_train_full[:, 1]
        print(min(house_age))
        print(max(house_age))
        avg_rooms = self.x_train_full[:, 2]
        print(min(avg_rooms))
        print(max(avg_rooms))
        avg_brooms = self.x_train_full[:, 3]
        print(min(avg_brooms))
        print(max(avg_brooms))
        pop = self.x_train_full[:, 4]
        print(min(pop))
        print(max(pop))
        avg_occup = self.x_train_full[:, 5]
        print(min(avg_occup))
        print(max(avg_occup))
        coordinates = self.x_train_full[:, -2:]
        target = self.y_train_full

        sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1],
                        size=target, hue=target, palette="viridis", alpha=0.5)
        plt.legend(title="Value", loc="upper right")
        plt.title("North-West position of houses")
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.show()


    def execute_part_b(self, extreme_training="YES_Please"):
        # Part (b) code here
        # chose if you want to do extreme training or not
        # Define hyperparameters and architectures to test
        if extreme_training == "YES_Please":
            hidden_units_list = [8, 16, 32, 64, 128, 256]
            hidden_layers_list = [1, 2, 3]
            batch_sizes = [128, 256]
            epochs = 300
        else:
            hidden_units_list = [32]
            hidden_layers_list = [2]
            batch_sizes = [128]
            epochs = 300

        results = []
        for batch_size in batch_sizes:
            for hidden_units in hidden_units_list:
                for hidden_layers in hidden_layers_list:
                    model, model_name = self.create_NNM_regression(hidden_units, hidden_layers)
                    train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(
                        model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs,
                        self.x_test, self.y_test, callback=False)

                    results.append([hidden_units, hidden_layers, train_loss, batch_size, val_error])

        # Print the results in a table
        print(" Hidden Units | Hidden Layers | batch size | Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}".format(result[0], result[1], result[3], result[2], result[4]))

    def execute_part_c(self, early_stopping):
        # Part (c) code here
        # the paramters with the best results are:
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300


        model, model_name = self.create_NNM_regression_ADAM(hidden_units, hidden_layers)
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test, callback=early_stopping)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("Adam", train_loss, val_error))

        model, model_name = self.create_NNM_regression_SGD(hidden_units, hidden_layers)
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test, callback=early_stopping)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("SGD", train_loss, val_error))

        model, model_name = self.create_NNM_regression_momSGD(hidden_units, hidden_layers)
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test, callback=early_stopping)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("momSGD", train_loss, val_error))

        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        results = []

        for learning_rate in learning_rates:
            model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, learning_rate)
            train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                                 batch_size, epochs, self.x_test, self.y_test)
            results.append([learning_rate, "no", train_loss, val_error])

        for learning_rate in learning_rates:
            lr_schedule = schedules.ExponentialDecay(
                learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
            model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, lr_schedule)
            train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                                 batch_size, epochs, self.x_test, self.y_test)
            results.append([learning_rate, "yes", train_loss, val_error])

        print(" Learning Rate | Scheduled | Train Error |  Val Error")
        for result in results:
            print("{}|{}|{}|{}".format(result[0], result[1], result[2], result[3]))

    def execute_part_c_with_optuna(self):
        # Part (c) code here with optuna approach
        num_trials = 100

        print("ADAM")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective_adam, n_trials=num_trials)
        best_params = study.best_params
        best_mse = study.best_value
        print("Best Hyperparameters for Adam:", best_params)
        print("Best MSE for Adam:", best_mse)

        print("========================================================================\n\n\n")

        print("SGD")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective_SGD, n_trials=num_trials)
        best_params = study.best_params
        best_mse = study.best_value
        print("Best Hyperparameters for SGD:", best_params)
        print("Best MSE for SGD:", best_mse)

        print("========================================================================\n\n\n")

        print("Momentum SGD")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective_mom_SGD, n_trials=num_trials)
        best_params = study.best_params
        best_mse = study.best_value
        print("Best Hyperparameters for Momentum SGD:", best_params)
        print("Best MSE for Momentum SGD:", best_mse)

    def execute_part_d(self):
        # Part (d) code here
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300
        learning_rate = 0.01
        lr_schedule = schedules.ExponentialDecay(
            learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
        )

        model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, lr_schedule)

        self.train_evaluate_model_with_val(
            model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs, self.x_test, self.y_test
        )

        train_loss, test_accuracy, loss = self.train_evaluate_model_whole_training_set(
            model, self.x_train_full, self.y_train_full, batch_size, epochs, model_name, self.x_test, self.y_test
        )

        print("Whole training set")
        print("Train Error")
        print("{}".format(loss))

    def execute_part_e(self):
        # Part (e) code here
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300

        # y_train[y_train < 2], y_test[y_test < 2] = 0, 0
        # y_train[y_train >= 2], y_test[y_test >= 2] = 1, 1

        model, model_name = self.create_NNM_classification_classification(hidden_units, hidden_layers)
        train_loss, test_accuracy = self.train_evaluate_model_classification(
            model, self.x_train, self.y_train, batch_size, epochs, model_name, self.x_test, self.y_test
        )
        print("Test Loss | Test accuracy")
        print("{}|{}".format(train_loss, test_accuracy))

    def create_NNM_regression(self, hidden_units, hidden_layers):
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

        return model, model_name

    def create_NNM_regression_learning_rate(self, hidden_units, hidden_layers, lr_schedule):
        model = Sequential(name="NNM_Regression_LR_Schedule")
        model.add(Dense(hidden_units, input_dim=self.x_train.shape[1], activation='relu'))
        for _ in range(hidden_layers - 1):
            model.add(Dense(hidden_units, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error', metrics=['mse'])
        return model, "NNM_Regression_LR_Schedule"

    def create_NNM_classification(self, hidden_units, hidden_layers):
        model = Sequential(name="NNM_Classification")
        model.add(Dense(hidden_units, input_dim=self.x_train.shape[1], activation='relu'))
        for _ in range(hidden_layers - 1):
            model.add(Dense(hidden_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model, "NNM_Classification"

    def objective_adam(self, trial):
        hidden_units = trial.suggest_int("hidden_units", 16, 128)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        epochs = trial.suggest_int("epochs", 50, 300)

        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers)
        _, _, val_error = self.train_evaluate_model_with_val(
            model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs, self.x_test, self.y_test
        )

        return val_error

    def objective_SGD(self, trial):
        hidden_units = trial.suggest_int("hidden_units", 16, 128)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        epochs = trial.suggest_int("epochs", 50, 300)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers)
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mse'])

        _, _, val_error = self.train_evaluate_model_with_val(
            model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs, self.x_test, self.y_test
        )

        return val_error

    def objective_mom_SGD(self, trial):
        hidden_units = trial.suggest_int("hidden_units", 16, 128)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        epochs = trial.suggest_int("epochs", 50, 300)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        momentum = trial.suggest_uniform("momentum", 0.0, 1.0)

        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers)
        model.compile(optimizer=momSGD(learning_rate=learning_rate, momentum=momentum),
                      loss='mean_squared_error', metrics=['mse'])

        _, _, val_error = self.train_evaluate_model_with_val(
            model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs, self.x_test,
            self.y_test
        )

        return val_error

    def train_evaluate_model_with_val(self, model, model_name, x_train, y_train, x_val, y_val, batch_size, epochs,
                                      x_test, y_test, callback=True):
        # Configure the model training procedure

        if callback == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

            history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                                verbose=0, callbacks=[early_stopping])
        else:
            history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                                verbose=0)

        train_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

        # plot results
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Evolution of training and validation error for model {}'.format(model_name) + f"_batch_{batch_size}")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='best')
        plt.show()
        name = (str(model_name) + '-' + str(batch_size))
        plt.savefig(f'figures/{name}.png')

        val_error = history.history['val_loss']
        train_loss = history.history['loss']

        return train_loss[-1], test_accuracy, val_error[-1]

    def train_evaluate_model_whole_training_set(self, model, x_train, y_train, batch_size, epochs, model_name, x_test,
                                                y_test):
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        loss = model.evaluate(x_test, y_test, verbose=0)[0]

        # Visualize the learning curves
        plt.plot(history.history['loss'], label='train_loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return None, None, loss

    def train_evaluate_model_classification(self, model, x_train, y_train, batch_size, epochs, model_name, x_test,
                                            y_test):
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )

        train_loss = history.history['loss'][-1]
        test_accuracy = history.history['accuracy'][-1]

        # Visualize the learning curves
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_loss'], label='test_loss')
        plt.plot(history.history['val_accuracy'], label='test_accuracy')
        plt.title('Training and Testing Loss & Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

        return train_loss, test_accuracy

    def create_NNM_regression_ADAM(self, hidden_units, hidden_layers):
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

    def create_NNM_regression_SGD(self, hidden_units, hidden_layers):
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

    def create_NNM_regression_momSGD(self, hidden_units, hidden_layers):
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

    def create_NNM_regression_learning_rate(self, hidden_units, hidden_layers, learning_rate):
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



if __name__ == '__main__':
    data_dict = pickle.load(open("california-housing-dataset.pkl", "rb"))
    model_instance = HousingRegressionModel(data_dict)

    # Choose which part to execute
#    model_instance.execute_part_a()
    # b) takes quite a long time, so we would not recommend to not run it and instead just set the parameter to False,
    # but if you insist you can set the parameter to YES_Please
 #   model_instance.execute_part_b(extreme_training="YES_Please")

    # executing once with and once without early stopping
    model_instance.execute_part_c(early_stopping=True)
    #model_instance.execute_part_c(early_stopping=False)


    #modle_instance.execute_part_c_with_optuna()

    # model_instance.execute_part_d()
    # model_instance.execute_part_e()
