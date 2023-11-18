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
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
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
            hidden_units_list = [128, 64, 32, 8]
            hidden_layers_list = [3, 2, 1]
            batch_sizes = [256, 128, 64, 16]
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
                    model, model_name = self.create_NNM_regression(hidden_units, hidden_layers, optimizer="adam")
                    train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(
                        model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs,
                        self.x_test, self.y_test)

                    results.append([hidden_units, hidden_layers, train_loss, batch_size, val_error])

        # Print the results in a table
        print(" Hidden Units | Hidden Layers | batch size | Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}".format(result[0], result[1], result[3], result[2], result[4]))

    def execute_part_c(self):
        # Part (c) code here
        # the paramters with the best results are:
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300


        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers, optimizer=Adam())
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("Adam", train_loss, val_error))

        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers, optimizer=SGD())
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("SGD", train_loss, val_error))

        model, model_name = self.create_NNM_regression(hidden_units, hidden_layers, optimizer=momSGD())
        train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                             batch_size, epochs, self.x_test, self.y_test)
        print(" Classifier | Train Error |  Val Error")
        print("{}|{}|{}".format("momSGD", train_loss, val_error))

        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        results = []

        # iterate over the learning rates and print the results of the best model, which in our case is adam see results
        # above
        for learning_rate in learning_rates:
            model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, learning_rate)
            model_name = model_name + f"_lr: {learning_rate}"
            train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                                 batch_size, epochs, self.x_test, self.y_test)
            results.append([learning_rate, "no", train_loss, val_error])
            print(f"{model_name}_trLoss: {train_loss}_valLoss: {val_error}_noSchedual_lr{learning_rate}")

        # with learning schedule
        for learning_rate in learning_rates:
            lr_schedule = schedules.ExponentialDecay(learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
            model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, lr_schedule)
            model_name = model_name + f"_lr: {learning_rate}"
            train_loss, test_accuracy, val_error = self.train_evaluate_model_with_val(model, model_name, self.x_train, self.y_train, self.x_val, self.y_val,
                                                                                 batch_size, epochs, self.x_test, self.y_test)
            results.append([learning_rate, "yes", train_loss, val_error])
            print(f"{model_name}_trLoss: {train_loss}_valLoss: {val_error}_schedual_lr{learning_rate}")


        print(" Learning Rate | Scheduled | Train Error |  Val Error")
        for result in results:
            print("{}|{}|{}|{}".format(result[0], result[1], result[2], result[3]))


    def execute_part_d(self):
        # Part (d) code here
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300
        learning_rate = 0.001

        model, model_name = self.create_NNM_regression_learning_rate(hidden_units, hidden_layers, learning_rate)

        self.train_evaluate_model_with_val(
            model, model_name, self.x_train, self.y_train, self.x_val, self.y_val, batch_size, epochs, self.x_test, self.y_test
        )

        train_loss, test_accuracy, loss = self.train_evaluate_model_whole_training_set(
            model, model_name, self.x_train_full, self.y_train_full, batch_size, epochs, self.x_test, self.y_test)

        print("Whole training set")
        print("Train Error")
        print("{}".format(loss))

    def execute_part_e(self):
        # Part (e) code here
        hidden_units = 32
        hidden_layers = 2
        batch_size = 128
        epochs = 300

        # Redefine target variables for binary classification
        self.y_train_full[self.y_train_full < 2] = 0
        self.y_train_full[self.y_train_full >= 2] = 1
        self.y_test[self.y_test < 2] = 0
        self.y_test[self.y_test >= 2] = 1


        model, model_name = self.create_NNM_classification_classification(hidden_units, hidden_layers)
        train_loss, test_accuracy = self.train_evaluate_model_classification(
            model, model_name, self.x_train_full, self.y_train_full, batch_size, epochs, self.x_test, self.y_test
        )


    def create_NNM_regression(self, hidden_units, hidden_layers, optimizer):
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
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

        return model, model_name

    def create_NNM_regression_learning_rate(self, hidden_units, hidden_layers, lr_schedule):
        model_name = "hl_{}_hu_{}_adam".format(hidden_layers, hidden_units)
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
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr_schedule), metrics=['mse'])

        return model, model_name

    def create_NNM_classification_classification(self, hidden_units, hidden_layers):
        model_name = "hl_{}_hu_{}_classifcation".format(hidden_layers, hidden_units)
        # Configure the model layers
        model = Sequential()
        # Input layer
        model.add(Dense(units=hidden_units, input_dim=8, kernel_initializer='normal', activation='tanh'))
        # Hidden layers
        for _ in range(hidden_layers):
            model.add(Dense(hidden_units, activation='tanh'))

        # Only one output layer since we have a regression task
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model, model_name


    def train_evaluate_model_with_val(self, model, model_name, x_train, y_train, x_val, y_val, batch_size, epochs,
                                      x_test, y_test):

        # Configure the model training procedure
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                            verbose=0, callbacks=[early_stopping])


        train_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

        # plot results
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('training and validation error: {}'.format(model_name) + f"_batch_{batch_size}")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='best')
        plt.show()
        name = (str(model_name) + '-' + str(batch_size))
        plt.savefig(f'figures/{name}.png')

        val_error = history.history['val_loss']
        train_loss = history.history['loss']

        return train_loss[-1], test_accuracy, val_error[-1]

    def train_evaluate_model_whole_training_set(self, model, model_name, x_train, y_train, batch_size, epochs, x_test, y_test):
        early_stopping = keras.callbacks.EarlyStopping(monitor='mse', patience=30, restore_best_weights=True)

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

        loss = history.history['loss']

        # Evaluate the final model on the test set
        test_error = mean_squared_error(y_test, y_pred)

        # Report the final test error
        print(f"Final Test Error: {test_error:.4f}")

        return train_loss, test_accuracy, loss[-1]

    def train_evaluate_model_classification(self, model, model_name, x_train, y_train, batch_size, epochs, x_test,
                                            y_test):
        early_stopping = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[early_stopping])

        # Evaluate the model on the test set
        y_pred_proba = model.predict(x_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Print evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision


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

        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=['mse'])

        return model, model_name



if __name__ == '__main__':
    data_dict = pickle.load(open("california-housing-dataset.pkl", "rb"))
    model_instance = HousingRegressionModel(data_dict)

    # Choose which part to execute
#    model_instance.execute_part_a()
    # b) takes quite a long time, so we would not recommend to not run it and instead just set the parameter to False,
    # but if you insist you can set the parameter to YES_Please
#    model_instance.execute_part_b(extreme_training="no")

    # executing once with and once without early stopping
#    model_instance.execute_part_c()


#  model_instance.execute_part_d()
    model_instance.execute_part_e()
