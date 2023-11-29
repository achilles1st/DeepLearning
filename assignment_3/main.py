import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.models import Sequential
from tensorflow.keras import layers, models, Input, Model
import tensorflow as tf
import random


class CNN:
    def __init__(self, train_data, test_data, perturb):
        # load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_train, self.x_test_ = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        # load perturbed dataset
        self.x_perturb, self.y_perturb = perturb["x_perturb"], perturb["y_perturb"]
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_perturb = np.mean(self.x_perturb, axis=3)

        # normalize to range 0-1
        self.x_train, self.x_test = self.preprocess_data(self.x_train, self.x_test)

        # one hot encode target values
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=20)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=20)

        # split train data into train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.25, random_state=42, stratify=self.y_train
        )

    def preprocess_data(self, train, test):

        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')

        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0

        # return normalized images
        return train_norm, test_norm

    def plot_samples(self, x_data, y_data, class_labels, num_samples=3):
        plt.figure(figsize=(20, 10))
        # loop over each class label and create a grid of num_samples x num_samples
        for i, label in enumerate(class_labels):
            class_indices = np.where(np.argmax(y_data, axis=1) == label)[0]
            samples = random.sample(list(class_indices), min(num_samples, len(class_indices)))
            # loop over each sample and plot it
            for j, idx in enumerate(samples):
                if num_samples == 1:
                    plt.subplot(5, 4, i + 1)
                else:
                    plt.subplot(len(class_labels), num_samples, i * num_samples + j + 1)

                plt.imshow(np.squeeze(x_data[idx]), cmap='gray')
                plt.title(f'Class {label}')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig('figures/classes.png')
        plt.show()

    def execute_a(self):

        print(self.x_train.shape)
        print(self.y_train.shape)

        print(self.x_test.shape)
        print(self.y_test.shape)
        # plot 3 random samples from each class (10 classes)
        self.plot_samples(self.x_train, self.y_train,
                          class_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                          num_samples=1)

    def execute_b(self):

        filters_list = [64, 32, 8]
        batch_sizes = [128, 64, 32, 16]
        epochs = 300

        # filters_list = [64]
        # hidden_layers_list = [2]
        # batch_sizes = [64]
        # epochs = 4
        # learning_rates = [0.001]
        # scheduled = [True]

        # iterate over all wanted values and build and train a coresponding model
        results = []
        for filters in filters_list:
            for batch_size in batch_sizes:
                model, modelname = self.create_CNN_classification_3_layers(filters)
                accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model(model,
                                                                                   modelname,
                                                                                   batch_size, epochs)
                results.append([filters, 3, batch_size, accuracy, val_accuracy, loss, val_loss])

                model, modelname = self.create_CNN_classification_2_layers(filters)
                accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model(model,
                                                                                   modelname,
                                                                                   batch_size, epochs)
                results.append([filters, 2, batch_size, accuracy, val_accuracy, loss, val_loss])

                model, modelname = self.create_CNN_classification_1_layer(filters)
                accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model(model,
                                                                                   modelname,
                                                                                   batch_size, epochs)
                results.append([filters, 1, batch_size, accuracy, val_accuracy, loss, val_loss])

        print("Filters | Hidden Layers | batch size |  Train Accuracy | Val Accuracy | Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}|{}|{}|".format(result[0], result[1], result[2], result[3], result[4],
                                                 result[5], result[6]))

    def create_CNN_classification_3_layers(self, filters):
        model_name = "hl_{}_fi_{}".format(2, filters)

        # input layer
        input_image = Input(shape=(32, 32, 1))

        # convolution layers
        l1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='VALID')(input_image)
        l1 = layers.MaxPooling2D((2, 2))(l1)

        l2 = layers.Conv2D(filters=filters * 2, kernel_size=(3, 3), activation='relu', padding='VALID')(l1)
        l2 = layers.MaxPooling2D((2, 2))(l2)

        l3 = layers.Conv2D(filters=filters * 4, kernel_size=(3, 3), activation='relu', padding='VALID')(l2)
        l3 = layers.MaxPooling2D((2, 2))(l3)

        l4 = layers.Flatten()(l3)
        l5 = layers.Dense(units=1024, activation='relu')(l4)
        l6 = layers.Dense(units=20, activation='relu')(l5)
        y_pred = layers.Dense(units=20)(l6)

        model = Model(input_image, y_pred)

        model.summary()

        lr_rates = [1e-3, 1e-4, 1e-5]
        lr_boundaries = [1000,1500]
        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_rates)

        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=lr_fn),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, model_name

    def create_CNN_classification_2_layers(self, filters):
        model_name = "hl_{}_fi_{}".format(1, filters)

        # input layer
        input_image = Input(shape=(32, 32, 1))

        # convolution layers
        l1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='VALID')(input_image)
        l1 = layers.MaxPooling2D((2, 2))(l1)

        l3 = layers.Conv2D(filters=filters * 4, kernel_size=(3, 3), activation='relu', padding='VALID')(l1)
        l3 = layers.MaxPooling2D((2, 2))(l3)

        l4 = layers.Flatten()(l3)
        l5 = layers.Dense(units=1024, activation='relu')(l4)
        l6 = layers.Dense(units=20, activation='relu')(l5)
        y_pred = layers.Dense(units=20)(l6)

        model = Model(input_image, y_pred)

        model.summary()

        lr_rates = [1e-3, 1e-4, 1e-5]
        lr_boundaries = [1000, 1500]
        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_rates)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fn),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, model_name

    def create_CNN_classification_1_layer(self, filters):
        model_name = "hl_{}_fi_{}".format(3, filters)

        # input layer
        input_image = Input(shape=(32, 32, 1))

        # convolution layers
        l1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='VALID')(input_image)
        l1 = layers.MaxPooling2D((2, 2))(l1)

        l4 = layers.Flatten()(l1)
        l5 = layers.Dense(units=1024, activation='relu')(l4)
        l6 = layers.Dense(units=20, activation='relu')(l5)
        y_pred = layers.Dense(units=20)(l6)

        model = Model(input_image, y_pred)

        model.summary()

        lr_rates = [1e-3, 1e-4, 1e-5]
        lr_boundaries = [1000, 1500]
        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_rates)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fn),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
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
