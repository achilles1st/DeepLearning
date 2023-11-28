import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.models import Sequential
from tensorflow.keras import layers, models
import tensorflow as tf


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

        hidden_units_list = [256, 128, 64, 32, 8]
        hidden_layers_list = [3, 2, 1]
        batch_sizes = [256, 128, 64, 16]
        epochs = 300
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        scheduled = [True, False]

        # hidden_units_list = [64]
        # hidden_layers_list = [2]
        # batch_sizes = [64]
        # epochs = 4
        # learning_rates = [0.001]
        # scheduled = [True]

        results = []
        for hidden_unit in hidden_units_list:
            for hidden_layer in hidden_layers_list:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        for schedule in scheduled:
                            model, modelname = self.create_CNN_classification(hidden_unit, hidden_layer,
                                                                              learning_rate, schedule)
                            accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model(model,
                                                                                               modelname,
                                                                                               batch_size, epochs)
                            results.append([hidden_unit, hidden_layer, batch_size, learning_rate, schedule, accuracy,
                                            val_accuracy, loss, val_loss])

        print("Hidden Units | Hidden Layers | batch size | Learning Rate | Schedule | Train Accuracy | Val Accuracy | "
              " Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}|{}|{}|{}|{}".format(result[0], result[1], result[2], result[3], result[4],
                                                      result[5], result[6], result[7], result[8]))

    def create_CNN_classification(self, hidden_units, hidden_layers, learning_rate, scheduled):
        model_name = "hl_{}_hu_{}_lr_{}_Sc_{}".format(hidden_layers, hidden_units, learning_rate, scheduled)
        # Configure the model layers
        model = Sequential()
        # Input layer
        if scheduled:
            learning_rate = schedules.ExponentialDecay(learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

        # for _ in range(hidden_layers):
        model = models.Sequential()

        for count, _ in enumerate(range(hidden_layers)):
            model.add(layers.Conv2D(hidden_units / (2 ** count), (3, 3), activation='relu', input_shape=(32, 32, 1),
                                    padding='VALID'))
            model.add(layers.MaxPooling2D((2, 2)))
            print(hidden_units / (2 ** count))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(20, activation='softmax'))  # Output layer with 20 classes

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        return model, model_name

    def train_evaluate_model(self, model, model_name, batch_size, epochs):
        name = (str(model_name) + '_Bs_' + str(batch_size))
        print(name)

        # Configure the model training procedure
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=16,
                                                          restore_best_weights=True)

        history = model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(self.x_val, self.y_val), callbacks=[early_stopping])
        # # plot results

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # First plot (top)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.set_title('Accuracy')
        ax1.legend(['train', 'validation'], loc='best')

        # Second plot (bottom)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.set_title('Loss')
        ax2.legend(['train', 'validation'], loc='best')
        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plots
        # plt.show()

        name = (str(model_name) + '-' + str(batch_size))
        plt.savefig(f'figures/{name}.png')

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        return accuracy[-1], val_accuracy[-1], loss[-1], val_loss[-1]


if __name__ == '__main__':
    dict = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, dict)

    # model_instance.execute_a()

    model_instance.execute_b()
