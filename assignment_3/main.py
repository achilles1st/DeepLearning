import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.models import Sequential
from tensorflow.keras import layers, models


class CNN:
    def __init__(self, train_data, test_data, dict):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        self.x_train, self.x_test = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        self.x_perturb, self.y_perturb = dict["x_perturb"], dict["y_perturb"]
        self.x_perturb = np.mean(self.x_perturb, axis=3)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.25, random_state=42, stratify=self.y_train
        )

    def execute_a(self):

        print(self.x_train.shape)
        print(self.y_train.shape)

        print(self.x_test.shape)
        print(self.y_test.shape)

        unique = []

        for i in range(20):
            unique.append(np.where(self.y_train == i)[0][0])

        plt.figure(figsize=(12, 8))
        for index in range(20):
            plt.subplot(5, 4, index + 1)
            plt.subplots_adjust(hspace=0.5)
            plt.imshow(self.x_train[unique[index]], cmap='gray')
            plt.title(self.y_train[unique[index]])
        plt.show()

    def execute_b(self):

        # hidden_units_list = [128, 64, 32, 8]
        # hidden_layers_list = [3, 2, 1]
        # batch_sizes = [256, 128, 64, 16]
        # epochs = 300
        # learning_rates = [0.1, 0.01, 0.001, 0.0001]
        # scheduled = [True, False]

        hidden_units_list = [64]
        hidden_layers_list = [2]
        batch_sizes = [64]
        epochs = 300
        learning_rates = [0.01]
        scheduled = [True]

        model, modelname = self.create_CNN_classification(hidden_units_list, hidden_layers_list, learning_rates, scheduled)

    def create_CNN_classification(self, hidden_units, hidden_layers, learning_rate, scheduled):
        model_name = "hl_{}_hu_{}".format(hidden_layers, hidden_units)
        # Configure the model layers
        model = Sequential()
        # Input layer
        if scheduled:
            learning_rate = schedules.ExponentialDecay(learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

        for _ in range(len(hidden_layers)):
            model.add(layers.Conv2D(hidden_units, (3, 3), activation='relu', input_shape=(32, 32, 1)))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(20, activation='sigmoid'))  # For binary classification

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        return model, model_name


if __name__ == '__main__':
    dict = pickle.load(open("C:/Users/nikla/Uni/Mit_Stef/DeepLearning/cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, dict)

    # model_instance.execute_a()

    model_instance.execute_b()
