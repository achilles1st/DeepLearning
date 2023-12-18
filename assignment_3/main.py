import tensorflow as tf
import keras.datasets as tfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.optimizers import schedules
from keras.models import Sequential
from keras import layers, models, Input, Model
from keras.regularizers import l1_l2
from keras.regularizers import l2
import tensorflow as tf
import random
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CNN:
    def __init__(self, train_data, test_data, perturb):
        # load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train_data, test_data
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_train, self.x_test = np.mean(self.x_train, axis=3), np.mean(self.x_test, axis=3)
        # load perturbed dataset
        self.x_perturb, self.y_perturb = perturb["x_perturb"], perturb["y_perturb"]
        # converting all images to grayscale via averaging the three RGB channel pixel values
        self.x_perturb = np.mean(self.x_perturb, axis=3)

        # split train data into train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.25, random_state=42, stratify=self.y_train)

        # normalize to range 0-1
        self.x_train, self.x_test, self.x_val = self.preprocess_data(self.x_train, self.x_test, self.x_val)

        # one hot encode target values
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=20)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=20)
        self.y_val = tf.keras.utils.to_categorical(self.y_val, num_classes=20)

        # get whole set
        (self.x_train_whole, self.y_train_whole), (self.x_test, self.y_test) = train_data, test_data
        self.x_train_whole = np.mean(self.x_train_whole, axis=3)
        self.x_train_whole = self.x_train_whole.astype('float32')
        self.x_train_whole = self.x_train_whole / 255.0
        self.y_train_whole = tf.keras.utils.to_categorical(self.y_train_whole, num_classes=20)




    def preprocess_data(self, train, test, x_val):

        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        val_norm = x_val.astype('float32')

        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        val_norm = val_norm / 255.0

        # return normalized images
        return train_norm, test_norm, val_norm

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
        #plt.savefig('/content/drive/MyDrive/figures/classes.png')
        plt.show()

    def execute_a(self):
        # print shapes of train and test data
        print(self.x_train.shape)
        print(self.y_train.shape)

        print(self.x_test.shape)
        print(self.y_test.shape)
        # plot 3 random samples from each class (20 classes)
        self.plot_samples(self.x_train, self.y_train,
                          class_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                          num_samples=1)

    def execute_b(self):

        filters_list = [8, 16, 32, 64]
        batch_sizes = [128, 64, 32, 16]
        hidden_layers_list = [1,2,3]
        epochs = 200

        #filters_list = [32]
        #hidden_layers_list = [3]
        #batch_sizes = [64]
        #epochs = 100
        #learning_rates = [0.001]

        # iterate over all wanted values and build and train a coresponding model
        results = []
        for filters in filters_list:
          print("Filters | Hidden Layers | batch size |  Train Accuracy | Val Accuracy | Train Error | Val Error")
          for result in results:
            print("{}|{}|{}|{}|{}|{}|{}|".format(result[0], result[1], result[2], result[3], result[4],
                                                 result[5], result[6]))
          for batch_size in batch_sizes:
              for layer in hidden_layers_list:
                  model, modelname = self.create_CNN_model(filters, layer)
                  accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model(model,
                                                                                      modelname,
                                                                                      batch_size, epochs)
                  results.append([filters, layer, batch_size, accuracy, val_accuracy, loss, val_loss])

        print("Filters | Hidden Layers | batch size |  Train Accuracy | Val Accuracy | Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}|{}|{}|".format(result[0], result[1], result[2], result[3], result[4],
                                                 result[5], result[6]))



    def execute_c(self):
        filters = 64
        layer = 3
        batch_size = 128
        epochs = 200

        dropout_list = [0.1, 0.2, 0.3, 0.5]
        l2_list = [0.1, 0.01, 0.001, 0.0001]


        # iterate over all wanted values and build and train a coresponding model
        results = []
        for dropout in dropout_list:
            for l2 in l2_list:
                model, modelname = self.create_CNN_model_regu_dropout(filters, layer, dropout, l2)
                accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model_regu(model,
                                                                                       modelname,
                                                                                       batch_size, epochs)
                results.append([dropout, l2, accuracy, val_accuracy, loss, val_loss])

        print("Dropout Rate | L2 |  Train Accuracy | Val Accuracy | Train Error | Val Error")
        for result in results:
            print("{}|{}|{}|{}|{}|{}|".format(result[0], result[1], result[2], result[3], result[4],
                                                 result[5]))

    def execute_d(self):
        filters = 64
        layer = 3
        batch_size = 128
        epochs = 5

        dropout = 0.1
        l2 = 0.0001

        results = []
        model, modelname = self.create_CNN_model_regu_dropout(filters, layer, dropout, l2)

        # accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model_regu(model, modelname, batch_size, epochs)
        accuracy, loss, test_loss , test_acc = self.train_evaluate_model_whole(model, modelname, batch_size, epochs)
        results.append([accuracy, loss, test_loss, test_acc])

        print("Train Accuracy | Train Error | Test Error | Test Acc")
        for result in results:
            print("{}|{}|{}|{}".format(result[0], result[1], result[2], result[3]))

    def execute_e(self):
        # parameters
        filters = 64
        layer = 3
        batch_size = 128
        epochs = 300

        dropout = 0.1
        l2 = 0.0001
        # to be iterated over
        l1_list = [0.1, 0.01, 0.001, 0.0001]
        batch_list = [True,False]
        augment_list = [True,False]


        model, modelname = self.create_CNN_model_regu_dropout(filters, layer, dropout, l2)

        accuracy, val_accuracy, loss, val_loss = self.train_evaluate_model_regu(model, modelname, batch_size, epochs)
        accuracy, loss, test_loss , test_acc = self.train_evaluate_model_whole(model, modelname, batch_size, epochs)

        perturb_loss, perturb_accuracy = model.evaluate(self.x_perturb, self.y_perturb, batch_size=batch_size)

        print("Perturb Accuracy | Perturb Error")
        print("{}|{}".format(perturb_loss, perturb_accuracy))


        # iterate over all wanted values and build and train a coresponding model
        results = []
        for l1 in l1_list:
            for batch in batch_list:
                for augment in augment_list:
                    model, modelname = self.create_CNN_model_regu_dropout_norm(filters, layer, dropout, l2, l1, batch)
                    accuracy, loss = self.train_evaluate_model_regu_augment(model, modelname, batch_size, epochs, augment)
                    results.append([l1, batch,augment, accuracy, loss])

        print("L1| Batch Normalization |  Augment | Train Accuracy | Train Error")
        for result in results:
            print("{}|{}|{}|{}|{}|{}|".format(result[0], result[1], result[2], result[3], result[4]))

    def create_CNN_model_regu_dropout(self, filters, layer, dropout, l2_regu):
        model_name = "hl_{}_fi_{}".format(layer, filters)

        # input layer
        model = models.Sequential()
        # create layerd model
        for count, _ in enumerate(range(layer)):
            model.add(layers.Conv2D(filters*(2 ** count), (3, 3), activation='relu', kernel_regularizer=l2(l2_regu), input_shape=(32, 32, 1),
                                  padding='VALID'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(Dropout(dropout))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_regu)))
        model.add(layers.Dense(20, activation='softmax'))  # Output layer with 20 classes

        model.summary()

        lr_rates = [1e-3, 1e-4, 1e-5]
        lr_boundaries = [1000, 1500]
        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_rates)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fn),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, model_name

    def create_CNN_model_regu_dropout_norm(self, filters, layer, dropout, l2_regu, l1_regu, BatchNorm):
        model_name = "hl_{}_fi_{}".format(layer, filters)

        # input layer
        model = models.Sequential()

        for count, _ in enumerate(range(layer)):
            model.add(layers.Conv2D(filters*(2 ** count), (3, 3), activation='relu', kernel_regularizer=l1_l2(l1=l1_regu, l2=l2_regu), input_shape=(32, 32, 1),
                                  padding='VALID'))
            if BatchNorm:
                model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(Dropout(dropout))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=l1_regu, l2=l2_regu)))
        if BatchNorm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(20, activation='softmax'))  # Output layer with 20 classes

        model.summary()

        lr_rates = [1e-3, 1e-4, 1e-5]
        lr_boundaries = [1000, 1500]
        lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_rates)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fn),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model, model_name



    def create_CNN_model(self, filters, layer):
        # create model name
        model_name = "hl_{}_fi_{}".format(layer, filters)

        model = models.Sequential()

        for count, _ in enumerate(range(layer)):
            model.add(layers.Conv2D(filters*(2 ** count), (3, 3), activation='relu', input_shape=(32, 32, 1),
                                    padding='VALID'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(20, activation='softmax'))  # Output layer with 20 classes

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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                          restore_best_weights=True)

        history = model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1,
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
        plt.show()

        name = (str(model_name) + '-' + str(batch_size))
        #plt.savefig(f'/content/drive/MyDrive/figures/{name}.png')

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        return accuracy[-1], val_accuracy[-1], loss[-1], val_loss[-1]

    def train_evaluate_model_regu(self, model, model_name, batch_size, epochs):
        name = (str(model_name) + '_Bs_' + str(batch_size))
        print(name)

        # Configure the model training procedure
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                          restore_best_weights=True)

        history = model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1,
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
        plt.show()

        name = (str(model_name) + '-' + str(batch_size))
        #plt.savefig(f'/content/drive/MyDrive/figures/{name}.png')

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        return accuracy[-1], val_accuracy[-1], loss[-1], val_loss[-1]


    def train_evaluate_model_regu_augment(self, model, model_name, batch_size, epochs, augment):
        name = (str(model_name) + '_Bs_' + str(batch_size))
        print(name)

        # Configure the model training procedure
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10,
                                                          restore_best_weights=True)

        history = model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                            validation_data=(self.x_val, self.y_val), callbacks=[early_stopping])

        if augment:
          sampler = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                                            width_shift_range=0.1, height_shift_range=0.1,
                                                            validation_split=0.1).flow(self.x_train, self.y_train, batch_size=batch_size)

          history = model.fit(sampler, epochs=epochs, batch_size=batch_size, verbose=1)


        perturb_loss, perturb_accuracy = model.evaluate(self.x_perturb, self.y_perturb, batch_size=batch_size)

        print("Perturb Accuracy | Perturb Error")
        print("{}|{}".format(perturb_loss, perturb_accuracy))


        accuracy = history.history['accuracy']
        loss = history.history['loss']


        return accuracy[-1], loss[-1]


    def train_evaluate_model_whole(self, model, model_name, batch_size, epochs):
        name = (str(model_name) + '_Bs_' + str(batch_size))
        self.x_test = np.mean(self.x_test, axis=3)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=20)

        # Configure the model training procedure
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10,
                                                          restore_best_weights=True)

        history = model.fit(self.x_train_whole, self.y_train_whole, epochs=epochs, batch_size=batch_size, verbose=1
                           , callbacks=[early_stopping])

        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, batch_size=batch_size)

        y_pred = model.predict(self.x_test)

        #Create confusion Matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        labels = [str(i) for i in range(20)]  # Update labels according to your classes
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()


        name = (str(model_name) + '-' + str(batch_size))
        #plt.savefig(f'/content/drive/MyDrive/figures/{name}.png')

        accuracy = history.history['accuracy']
        loss = history.history['loss']

        return accuracy[-1], loss[-1], test_loss, test_accuracy

if __name__ == '__main__':
    perturb = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")
    model_instance = CNN(train_data, test_data, perturb)
    model_instance.execute_a()
    model_instance.execute_b()
    model_instance.execute_c()
    model_instance.execute_d()
    model_instance.execute_e()
